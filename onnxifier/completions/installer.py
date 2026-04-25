"""Install shell completions for onnxify.

This module is designed to be importable without pulling in heavy dependencies
such as onnx, so that ``onnxify --install-completion`` works even when the
runtime environment is not fully set up for model conversion.
"""

import locale
import subprocess
import sys
from pathlib import Path


def _completion_dir() -> Path:
    return Path(__file__).parent.resolve()


def install_bash() -> None:
    """Append the bash completion source line to ~/.bashrc."""
    bashrc = Path.home() / ".bashrc"
    script = _completion_dir() / "onnxify-completion.bash"
    marker = "# >>> onnxify bash completion >>>"
    end_marker = "# <<< onnxify bash completion <<<"
    block = f'{marker}\nsource "{script}"\n{end_marker}'

    if bashrc.exists():
        content = bashrc.read_text(encoding="utf-8")
        if marker in content:
            print("Bash completion is already installed in ~/.bashrc")
            return

    with open(bashrc, "a", encoding="utf-8") as f:
        f.write(f"\n{block}\n")
    print(f"Installed bash completion to {bashrc}")
    print("Run 'source ~/.bashrc' or restart your shell to activate.")


def _decode_pwsh_output(raw: bytes) -> str:
    """Decode raw bytes from pwsh, trying likely encodings."""
    _enc = ["utf-8", "gbk", "cp936"]
    for enc in set([locale.getpreferredencoding(False), *_enc]):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    # Last resort: replace errors
    return raw.decode("utf-8", errors="replace").strip()


def install_pwsh() -> None:
    """Append the PowerShell completion source line to $PROFILE."""
    profile_path: Path | None = None

    # 1. Try to ask pwsh/powershell where its profile is.
    for shell in ("pwsh", "powershell"):
        try:
            result = subprocess.run(
                [shell, "-NoProfile", "-Command", "Write-Host $PROFILE"],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                continue
            candidate = _decode_pwsh_output(result.stdout)
            if candidate:
                p = Path(candidate)
                if p.exists():
                    profile_path = p
                    break
        except FileNotFoundError:
            continue

    # 2. Fallback to well-known paths.
    if profile_path is None:
        if sys.platform == "win32":
            candidates = [
                Path.home()
                / "Documents"
                / "PowerShell"
                / "Microsoft.PowerShell_profile.ps1",
                Path.home()
                / "Documents"
                / "WindowsPowerShell"
                / "Microsoft.PowerShell_profile.ps1",
            ]
        else:
            candidates = [
                Path.home()
                / ".config"
                / "powershell"
                / "Microsoft.PowerShell_profile.ps1",
                Path.home() / ".powershell" / "Microsoft.PowerShell_profile.ps1",
            ]
        for c in candidates:
            if c.exists():
                profile_path = c
                break
        if profile_path is None:
            # Default to the first candidate, creating parent dirs later.
            profile_path = candidates[0]

    script = _completion_dir() / "onnxify-completion.ps1"
    marker = "# >>> onnxify pwsh completion >>>"
    end_marker = "# <<< onnxify pwsh completion <<<"
    block = f'{marker}\n. "{script}"\n{end_marker}'

    if profile_path.exists():
        content = profile_path.read_text(encoding="utf-8")
        if marker in content:
            print(f"PowerShell completion is already installed in {profile_path}")
            return

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "a", encoding="utf-8") as f:
        f.write(f"\n{block}\n")
    print(f"Installed PowerShell completion to {profile_path}")
    print("Run '. $PROFILE' or restart PowerShell to activate.")


def install(shell: str) -> None:
    """Install completion for the given shell."""
    shell = shell.lower()
    if shell in ("bash",):
        install_bash()
    elif shell in ("pwsh", "powershell", "ps"):
        install_pwsh()
    else:
        raise ValueError(f"Unsupported shell: {shell}. Supported values: bash, pwsh")
