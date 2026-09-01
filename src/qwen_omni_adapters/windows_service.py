from __future__ import annotations

import sys

from qwen_omni_adapters.daemon import DaemonConfig, OmniDaemon


def main() -> int:
    if sys.platform != "win32":
        print("The Windows Service wrapper is available only on Windows.", file=sys.stderr)
        return 2
    try:
        import servicemanager  # type: ignore[import-not-found]
        import win32event  # type: ignore[import-not-found]
        import win32service  # type: ignore[import-not-found]
        import win32serviceutil  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"pywin32 is required for Windows Service mode: {exc}", file=sys.stderr)
        return 2

    class QwenOmniService(win32serviceutil.ServiceFramework):
        _svc_name_ = "QwenOmniAdapters"
        _svc_display_name_ = "Qwen Omni Adapters"
        _svc_description_ = "Supervises the logical Qwen Omni Ollama adapter and local dashboard."

        def __init__(self, args):
            super().__init__(args)
            self.stop_handle = win32event.CreateEvent(None, 0, 0, None)
            self.supervisor: OmniDaemon | None = None

        def SvcStop(self):  # noqa: N802 - pywin32 service ABI
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            if self.supervisor is not None:
                self.supervisor.request_stop()
            win32event.SetEvent(self.stop_handle)

        def SvcDoRun(self):  # noqa: N802 - pywin32 service ABI
            servicemanager.LogInfoMsg("Qwen Omni Adapters service starting")
            config = DaemonConfig.from_environment(allow_direct_gpu=True)
            self.supervisor = OmniDaemon(config)
            try:
                self.supervisor.run(register_signals=False)
            except Exception as exc:  # noqa: BLE001 - service manager needs a terminal event
                servicemanager.LogErrorMsg(f"Qwen Omni Adapters failed: {exc}")
                raise
            finally:
                servicemanager.LogInfoMsg("Qwen Omni Adapters service stopped")

    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(QwenOmniService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(QwenOmniService)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
