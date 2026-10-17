"""Bound temporary X11 key mappings to one Unicode typing attempt.

Only unused, unpressed, non-modifier codes are reserved. Numeric XTEST keycodes
avoid xdotool's scratch-symbol reuse. The caller holds this context until the
recipient has read the complete insertion, or its bounded readback expires.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
from contextlib import contextmanager


class KeymapUnavailable(RuntimeError):
    pass


class _Modifiers(C.Structure):
    _fields_ = [('max_keypermod', C.c_int), ('modifiermap', C.POINTER(C.c_ubyte))]


class UnicodeKeymap:
    def __init__(self, text, environment):
        self.characters = list(dict.fromkeys(c for c in text if ord(c) >= 128))
        if any(0xD800 <= ord(c) <= 0xDFFF for c in self.characters):
            raise KeymapUnavailable('native_unicode_character_unavailable')
        self.environment = environment
        self.display = None
        self.codes = {}
        self.assigned = {}

    def _bind(self):
        library = ctypes.util.find_library('X11')
        if not library or not self.environment.get('DISPLAY'):
            raise KeymapUnavailable('native_unicode_display_unavailable')
        self.lib = C.CDLL(library)
        signatures = {
            'XOpenDisplay': ([C.c_char_p], C.c_void_p),
            'XCloseDisplay': ([C.c_void_p], C.c_int),
            'XDisplayKeycodes': ([C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int)], C.c_int),
            'XGetKeyboardMapping': ([C.c_void_p, C.c_ubyte, C.c_int, C.POINTER(C.c_int)], C.POINTER(C.c_ulong)),
            'XChangeKeyboardMapping': ([C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_ulong), C.c_int], C.c_int),
            'XGetModifierMapping': ([C.c_void_p], C.POINTER(_Modifiers)),
            'XFreeModifiermap': ([C.POINTER(_Modifiers)], C.c_int),
            'XQueryKeymap': ([C.c_void_p, C.POINTER(C.c_ubyte)], C.c_int),
            'XGrabServer': ([C.c_void_p], C.c_int),
            'XUngrabServer': ([C.c_void_p], C.c_int),
            'XSync': ([C.c_void_p, C.c_int], C.c_int),
            'XFree': ([C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.display = self.lib.XOpenDisplay(self.environment['DISPLAY'].encode())
        if not self.display:
            raise KeymapUnavailable('native_unicode_display_unavailable')

    @contextmanager
    def _atomic(self):
        self.lib.XGrabServer(self.display)
        try:
            yield
        finally:
            self.lib.XUngrabServer(self.display)
            self.lib.XSync(self.display, 0)

    def _mapping(self):
        low, high, width = C.c_int(), C.c_int(), C.c_int()
        self.lib.XDisplayKeycodes(self.display, C.byref(low), C.byref(high))
        if not 8 <= low.value <= high.value <= 255:
            raise KeymapUnavailable('native_unicode_keymap_unavailable')
        pointer = self.lib.XGetKeyboardMapping(self.display, low.value, high.value-low.value+1, C.byref(width))
        if not pointer:
            raise KeymapUnavailable('native_unicode_keymap_unavailable')
        try:
            if not 1 <= width.value <= 32:
                raise KeymapUnavailable('native_unicode_keymap_unavailable')
            return {code: tuple(pointer[(code-low.value)*width.value+i] for i in range(width.value))
                    for code in range(low.value, high.value+1)}
        finally:
            self.lib.XFree(pointer)

    def _set(self, code, values):
        row = (C.c_ulong * len(values))(*values)
        self.lib.XChangeKeyboardMapping(self.display, code, len(values), row, 1)

    def __enter__(self):
        if not self.characters:
            return self
        try:
            self._bind()
            with self._atomic():
                mapping = self._mapping()
                modifiers = self.lib.XGetModifierMapping(self.display)
                if not modifiers:
                    raise KeymapUnavailable('native_unicode_modifiers_unavailable')
                try:
                    n = modifiers.contents.max_keypermod
                    if not 0 <= n <= 32:
                        raise KeymapUnavailable('native_unicode_modifiers_unavailable')
                    excluded = {modifiers.contents.modifiermap[i] for i in range(8*n)}
                finally:
                    self.lib.XFreeModifiermap(modifiers)
                pressed = (C.c_ubyte * 32)()
                self.lib.XQueryKeymap(self.display, pressed)
                unused = [code for code, row in mapping.items() if not any(row)
                          and code not in excluded and not pressed[code//8] & (1 << (code % 8))]
                if len(unused) < len(self.characters):
                    raise KeymapUnavailable('native_unicode_keymap_capacity')
                for char, code in zip(self.characters, unused, strict=False):
                    symbol = ord(char) if ord(char) < 256 else 0x01000000 | ord(char)
                    # Explicit levels prevent XKB from synthesizing upper-case
                    # symbols that were not part of this temporary assignment.
                    row = (symbol,) * len(mapping[code])
                    self.assigned[code] = row
                    self._set(code, row)
                    self.codes[char] = code
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.display:
            return
        conflict = False
        try:
            with self._atomic():
                current = self._mapping()
                for code, expected in self.assigned.items():
                    actual = current.get(code, ())
                    # XKB may expand a one-level mapping across columns. Every
                    # nonzero symbol must still be ours before restoring it.
                    if not actual or actual[0] != expected[0] or any(x not in (0, expected[0]) for x in actual):
                        conflict = True
                        continue
                    self._set(code, (0,) * len(actual))
        finally:
            self.lib.XCloseDisplay(self.display)
            self.display = None
        if conflict:
            raise KeymapUnavailable('native_unicode_keymap_changed')

    def __exit__(self, *_):
        self.close()
