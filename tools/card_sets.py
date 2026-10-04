"""Development card sets; these names never enter strategy decisions."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CARD_SETS = {"official": ("alpha", "beta", "gamma", "delta"),
             "examples": ("L1", "L2", "L3", "L4")}


def card_path(name):
    family = "official-cards" if name in CARD_SETS["official"] else "local-cards"
    if name not in sum(CARD_SETS.values(), ()):
        raise ValueError("unknown development card")
    return ROOT / ".local" / family / name


def card_order(names):
    names = tuple(names)
    for cards in CARD_SETS.values():
        if set(names) == set(cards):
            return cards
    raise ValueError("exactly one complete, recognized four-card set is required")


def require_runnable(name):
    root = card_path(name)
    config = root / "config/v4_scenario.json"
    if not config.is_file():
        raise ValueError("card files missing; run tools/prepare_official.py or tools/prepare_local.py")
    scenario = json.loads(config.read_text(encoding="utf-8"))
    products = [(value, (root / "config" / value).resolve()) for value in scenario["products"].values()]
    if any(not path.is_relative_to(root.resolve()) for _, path in products):
        raise ValueError("card product path escapes its directory")
    missing = [value for value, path in products if not path.is_file()]
    if missing:
        raise ValueError(name + " cannot run locally: missing official products " + ", ".join(missing))
