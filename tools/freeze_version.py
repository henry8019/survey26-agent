"""Save an immutable, credential-free evaluation project."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def freeze(destination, source=ROOT):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("version already exists")
    destination.mkdir(parents=True)
    for name in ("agent.py", "observer.project.json", "requirements.txt", "LICENSE", "LICENSE.md"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "agent_core", destination / "agent_core", ignore=shutil.ignore_patterns("__pycache__"))
    files = [destination / "agent.py", destination / "observer.project.json", *sorted((destination / "agent_core").glob("*.py"))]
    hashes = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (destination / "version.json").write_text(json.dumps({"source_sha256": fingerprint, "source_files": hashes}, indent=2), encoding="utf-8")
    return fingerprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(freeze(args.destination))
