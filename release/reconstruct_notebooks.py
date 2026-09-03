from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = [HERE / f"analysis_notebooks.tar.xz.part{i:02d}" for i in range(4)]
ARCHIVE = HERE / "analysis_notebooks.tar.xz"
EXPECTED_SHA256 = "b359e06455d0af4a7f4ece631b1d1734d7dec6f4b4a9c42e72e63f3bc563bab8"

missing = [path.name for path in PARTS if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing release part(s): {', '.join(missing)}")

with ARCHIVE.open("wb") as out:
    for part in PARTS:
        out.write(part.read_bytes())

actual_sha256 = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise RuntimeError(
        "Archive checksum mismatch. "
        f"Expected {EXPECTED_SHA256}, got {actual_sha256}."
    )

with tarfile.open(ARCHIVE, mode="r:xz") as tf:
    tf.extractall(HERE.parent)

print("Reconstructed and extracted sanitized notebooks into the repository root.")
print(f"Verified SHA-256: {actual_sha256}")
