"""Regression test for issue #3878: the .ps1 probe in get_windows_executable_command."""

import shutil

import pytest

from mcp.os.win32 import utilities


def test_command_that_only_exists_as_a_ps1_script_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolver never returns a .ps1 path, which could not be launched.

    Pins the fix for issue #3878: Windows shutil.which cannot resolve .ps1 (it is absent
    from PATHEXT), and passing one to a process spawn raises WinError 193.
    """
    probes: list[str] = []

    def which(command: str) -> str | None:
        probes.append(command)
        return r"C:\tools\pyserver.ps1" if command.endswith(".ps1") else None

    monkeypatch.setattr(shutil, "which", which)

    assert utilities.get_windows_executable_command("pyserver") == "pyserver"
    assert "pyserver.ps1" not in probes


def test_command_found_by_its_bare_name_is_returned_as_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An executable already on PATH resolves without an extension probe."""

    def which(command: str) -> str | None:
        return r"C:\Windows\System32\where.exe" if command == "where" else None

    monkeypatch.setattr(shutil, "which", which)

    assert utilities.get_windows_executable_command("where") == r"C:\Windows\System32\where.exe"


def test_command_found_through_a_script_extension_returns_that_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A .cmd shim resolves to the shim path, as the extension loop intends."""

    def which(command: str) -> str | None:
        return r"C:\tools\pyserver.cmd" if command.endswith(".cmd") else None

    monkeypatch.setattr(shutil, "which", which)

    assert utilities.get_windows_executable_command("pyserver") == r"C:\tools\pyserver.cmd"


def test_failed_probe_returns_the_bare_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OSError during PATH probing falls back to the command as given."""

    def which(command: str) -> str:
        raise OSError("probe failed")

    monkeypatch.setattr(shutil, "which", which)

    assert utilities.get_windows_executable_command("pyserver") == "pyserver"
