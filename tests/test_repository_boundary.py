from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED_DIRECTORIES = {"data", "figures", "manuscript", "outputs", "results"}
BANNED_SUFFIXES = {".aux", ".csv", ".dll", ".jpg", ".jpeg", ".log", ".pdf", ".png", ".so"}


def test_tracked_files_stay_within_the_public_code_boundary() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    violations = [
        path
        for path in tracked
        if path.parts[0] in BANNED_DIRECTORIES or path.suffix.lower() in BANNED_SUFFIXES
    ]
    assert not violations, f"non-code artifacts are tracked: {violations}"
