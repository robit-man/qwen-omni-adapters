from scripts.patch_kestrel_jetpack import MAIA_CASES, patch_source


def test_kestrel_jetpack_patch_guards_both_newer_torch_enums() -> None:
    source = "before\n" + "\nmiddle\n".join(MAIA_CASES) + "\nafter\n"

    patched, changed = patch_source(source)
    repeated, repeated_changed = patch_source(patched)

    assert changed == 2
    assert repeated_changed == 0
    assert repeated == patched
    assert patched.count("TORCH_VERSION_MINOR >= 7") == 2
