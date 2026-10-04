"""Download the published alpha-delta files using the official site's exporter.

Only public practice slugs are permitted. Weather/event files go to the simulator
under ignored .local, never to the agent or submission. Existing files are checked
and never silently replaced. The public storage credential is kept in memory.
"""
import argparse
import base64
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from card_sets import CARD_SETS, ROOT

SITE = "https://create.gosim.org"
RESOURCE = SITE + "/survey26/platform/resources"


def fetch(url, headers=None, data=None):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def public_storage():
    html = fetch(RESOURCE).decode()
    main_path = re.search(r'src="([^"]*/assets/index-[^"]+\.js)"', html).group(1)
    main = fetch(SITE + main_path).decode()
    module_path = re.search(r'"(?:\./)?(supabase-[^"/]+\.js)"', main).group(1)
    module = fetch(SITE + main_path.rsplit("/", 1)[0] + "/" + module_path).decode()
    hosts = set(re.findall(r'https://[a-z0-9]+\.supabase\.co', module))
    for token in re.findall(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', module):
        try:
            segment = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
        except (ValueError, KeyError):
            continue
        host = "https://" + claims.get("ref", "") + ".supabase.co"
        if claims.get("role") == "anon" and host in hosts:
            return host, {"apikey": token, "Authorization": "Bearer " + token}
    raise RuntimeError("official public exporter changed; no files downloaded")


def verify_manifest(destination):
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    for card in CARD_SETS["official"]:
        for name, digest in manifest["cards"][card]["files"].items():
            path = destination / card / name
            if not re.fullmatch(r"(?:config|public|truth)/[A-Za-z0-9_.-]+", name) or ".." in name:
                raise ValueError("unsafe manifest path")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("official resource changed or missing: " + card + "/" + name)
    return manifest


def prepare(destination, refresh=False):
    if (destination / "manifest.json").exists():
        existing = verify_manifest(destination)
        if not refresh:
            return existing
    host, auth = public_storage()
    manifest = {"source": RESOURCE, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "cards": {}}
    for card in CARD_SETS["official"]:
        slug = "v4-practice-" + card
        files = {}
        for directory in ("config", "public", "truth"):
            entries = json.loads(fetch(host + "/storage/v1/object/list/scenarios",
                                      {**auth, "Content-Type": "application/json"},
                                      json.dumps({"prefix": slug + "/" + directory, "limit": 1000,
                                                  "offset": 0, "sortBy": {"column": "name", "order": "asc"}}).encode()))
            if not isinstance(entries, list) or len(entries) >= 1000:
                raise RuntimeError("unexpected official directory listing")
            for entry in entries:
                name = entry.get("name", "")
                if entry.get("id") is None or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or ".." in name:
                    continue
                relative = directory + "/" + name
                data = fetch(host + "/storage/v1/object/scenarios/" + slug + "/" + relative, auth)
                target = destination / card / relative
                if target.exists() and target.read_bytes() != data:
                    raise ValueError("existing resource differs; left unchanged: " + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(data)
                files[relative] = hashlib.sha256(data).hexdigest()
        required = {"config/v4_scenario.json", "config/v4_score_config.json", "config/v4_fiber_config.json"}
        if not required <= files.keys():
            raise RuntimeError("missing official configuration: " + card)
        scenario = json.loads((destination / card / "config/v4_scenario.json").read_text(encoding="utf-8"))
        card_root = (destination / card).resolve()
        products = [(card_root / "config" / value).resolve().relative_to(card_root).as_posix()
                    for value in scenario["products"].values()]
        missing = sorted(set(products) - files.keys())
        identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        manifest["cards"][card] = {"slug": slug, "files": files, "sha256": identity,
                                    "missing_products": missing, "runnable": not missing,
                                    "source": SITE + "/survey26/platform/cards/" + card}
        print(card, len(files), "files", identity, flush=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return verify_manifest(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify local export without network access")
    parser.add_argument("--refresh", action="store_true", help="check for newly published files; reject changed existing files")
    args = parser.parse_args()
    if args.verify and args.refresh:
        parser.error("--verify and --refresh are mutually exclusive")
    destination = ROOT / ".local/official-cards"
    manifest = verify_manifest(destination) if args.verify else prepare(destination, args.refresh)
    print("Verified official alpha-delta export:", len(manifest["cards"]), "cards")
    for card, entry in manifest["cards"].items():
        if not entry["runnable"]:
            print(card, "cannot run locally; official export lacks:", ", ".join(entry["missing_products"]))


if __name__ == "__main__":
    main()
