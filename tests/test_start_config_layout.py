from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def _install_start_script(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "start.sh"
    install = tmp_path / "install"
    install.mkdir()
    script = install / "start.sh"
    shutil.copy2(source, script)
    script.chmod(0o755)

    python = install / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$CITRA_CONFIG_PATH"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    return install


def test_start_uses_split_config_directory(tmp_path: Path) -> None:
    install = _install_start_script(tmp_path)
    config_dir = install / ".citra" / "config"
    config_dir.mkdir(parents=True)
    for name in ("tools.toml", "models.toml"):
        (config_dir / name).write_text("", encoding="utf-8")

    result = subprocess.run(
        [str(install / "start.sh")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(config_dir)
    assert result.stderr == ""


def test_start_warns_and_supports_legacy_single_file(tmp_path: Path) -> None:
    install = _install_start_script(tmp_path)
    legacy = install / ".citra" / "config.toml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("", encoding="utf-8")

    result = subprocess.run(
        [str(install / "start.sh")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(legacy)
    assert "legacy Citra config detected" in result.stderr


def test_start_allows_split_config_without_global_linting(tmp_path: Path) -> None:
    install = _install_start_script(tmp_path)
    config_dir = install / ".citra" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "tools.toml").write_text("", encoding="utf-8")
    (config_dir / "models.toml").write_text("", encoding="utf-8")

    result = subprocess.run(
        [str(install / "start.sh")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(config_dir)
    assert result.stderr == ""


def test_start_rejects_missing_required_split_config(tmp_path: Path) -> None:
    install = _install_start_script(tmp_path)
    config_dir = install / ".citra" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "tools.toml").write_text("", encoding="utf-8")

    result = subprocess.run(
        [str(install / "start.sh")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert str(config_dir / "models.toml") in result.stderr
