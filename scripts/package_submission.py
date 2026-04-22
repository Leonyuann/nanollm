from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile


EXCLUDED_TOP_LEVEL_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "data",
    "outputs",
}
EXCLUDED_SUFFIXES = {".pdf", ".pt", ".pyc"}


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for submission packaging.

    Args:
        None.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Create a code-only submission zip archive.",
    )
    parser.add_argument(
        "--output",
        default="code.zip",
        help="Destination zip path.",
    )
    return parser


def should_exclude(path: Path, output_relative_path: Path | None) -> bool:
    """Decide whether a path should be excluded from the archive.

    Args:
        path: Candidate repository path.
        output_relative_path: Relative archive path when the archive is inside the repository.

    Returns:
        ``True`` when the path should be excluded.
    """
    if output_relative_path is not None and path == output_relative_path:
        return True
    if any(part in {".codex", ".vscode", "__pycache__"} for part in path.parts):
        return True
    if path.parts and path.parts[0] in EXCLUDED_TOP_LEVEL_DIRS:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    if path.name == ".DS_Store":
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Create a repository submission archive.

    Args:
        argv: Optional CLI-style argument sequence.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_relative_path = output_path.relative_to(repo_root)
    except ValueError:
        output_relative_path = None

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(repo_root.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(repo_root)
            if should_exclude(relative_path, output_relative_path):
                continue
            archive.write(path, arcname=relative_path)

    print(f"archive_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
