from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED_DIRECTORIES = {"data", "figures", "manuscript", "outputs", "results", "downloads", ".aris", "archive", "build", "dist"}
BANNED_SUFFIXES = {".aux", ".csv", ".dll", ".dylib", ".eps", ".jpg", ".jpeg",
                   ".json", ".log", ".npy", ".npz", ".parquet", ".pdf", ".pkl",
                   ".png", ".so", ".svg", ".tif", ".tiff", ".o", ".obj", ".exe", ".whl", ".gz", ".zip"}


def test_tracked_files_stay_within_the_public_code_boundary() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
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
    assert not violations, f"non-code artifacts are tracked or eligible for addition: {violations}"
