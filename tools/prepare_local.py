"""Download the pinned official runner/cards into ignored .local directories.

Uses UPSTREAM.json's exact archive SHA-256. Existing changed resources are never
overwritten. This prepares development tests, not the platform runtime.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="optional already downloaded official ZIP")
    args = parser.parse_args()
    upstream = json.loads((ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    local = ROOT / ".local"
    local.mkdir(exist_ok=True)
    archive = args.archive or local / "official-examples.zip"
    if not archive.exists():
        if args.archive:
            parser.error("--archive does not exist")
        temporary = local / "official-examples.zip.part"
        with urllib.request.urlopen(upstream["source_url"], timeout=30) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if digest(temporary) != upstream["archive_sha256"]:
            raise SystemExit("official archive checksum mismatch; no files installed")
        temporary.replace(archive)
    if digest(archive) != upstream["archive_sha256"]:
        raise SystemExit("official archive checksum mismatch; no files installed")
    files = []
    with zipfile.ZipFile(archive) as z:
        prefixes = ("gosim-observer-examples/runner/", "gosim-observer-examples/local-cards/")
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(prefixes):
                continue
            relative = PurePosixPath(info.filename).relative_to("gosim-observer-examples")
            if relative.is_absolute() or ".." in relative.parts or "\\" in info.filename:
                raise SystemExit("unsafe archive path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise SystemExit("archive symlink is not supported")
            destination = local.joinpath(*relative.parts).resolve()
            if not destination.is_relative_to(local.resolve()):
                raise SystemExit("archive path escapes .local")
            data = z.read(info)
            if destination.exists() and destination.read_bytes() != data:
                raise SystemExit(f"existing resource differs; left unchanged: {relative}")
            files.append((destination, data))
        if not any(p.name == "run_local.py" for p, _ in files):
            raise SystemExit("official runner missing from archive")
    for destination, data in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
    print(f"Prepared {len(files)} checksum-verified official resource files in .local")
    return subprocess.call([sys.executable, str(local / "runner/verify_engine.py")], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
