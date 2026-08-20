from __future__ import annotations

import argparse
from pathlib import Path


UNIT_NAMES = ("airaware-refresh.service", "airaware-refresh.timer", "airaware-api.service")


def install_units(repository_root, destination):
    repository_root = repository_root.resolve()
    if "\n" in str(repository_root) or "\r" in str(repository_root):
        raise ValueError("repository path must not contain a newline")
    source = repository_root / "deploy" / "systemd"
    destination.mkdir(parents=True, exist_ok=True)
    installed = []
    for name in UNIT_NAMES:
        content = (source / name).read_text(encoding="utf-8").replace("@REPO_ROOT@", str(repository_root))
        path = destination / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o644)
        installed.append(path)
    return installed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path.home() / ".config/systemd/user")
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    if not (repository_root / ".venv/bin/python").is_file():
        parser.error(f"virtual environment Python not found: {repository_root / '.venv/bin/python'}")
    for path in install_units(repository_root, args.destination):
        print(path)
    print("Create ~/.config/airaware/airaware.env with mode 600, then run:")
    print("systemctl --user daemon-reload")
    print("systemctl --user enable --now airaware-refresh.timer")
    print("systemctl --user enable --now airaware-api.service")


if __name__ == "__main__":
    main()
