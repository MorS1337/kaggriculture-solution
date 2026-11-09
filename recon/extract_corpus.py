#!/usr/bin/env python3
"""Strip the daily top-episodes dump down to what we actually need.

The dumps are ~32 MB of JSON per episode because they record full game state at
every step. We only need the actions, the seed and the outcome -- about 0.1% of
the bytes. This turns 21.5 GB into something that fits in memory.

The seed matters more than it looks: `info.seed` is recorded in every replay, so
any real ladder game can be reproduced locally bit-for-bit and replayed with our
agent substituted into either seat.

Output: one gzipped JSON per episode in corpus/actions/, plus an index.csv.

    python recon/extract_corpus.py corpus/d0829/kaggriculture-episodes-2026-08-29.zip
"""

import csv
import gzip
import hashlib
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "corpus" / "actions"


def _digest(items):
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(it, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()[:16]


def one(args):
    zip_path, name = args
    try:
        with zipfile.ZipFile(zip_path) as z:
            d = json.loads(z.read(name))
    except Exception as e:  # a truncated member should not kill the run
        return {"file": name, "error": repr(e)[:120]}

    info = d.get("info", {})
    steps = d["steps"]
    # The action an agent chose on turn t is recorded on steps[t+1], not steps[t]:
    # steps[0] is the initial state and its `action` field is the schema default.
    # Reading it off by one silently produces a plausible-looking but wrong tape --
    # verified by round-tripping a locally generated episode (recon/replay_check.py).
    acts = [[steps[t + 1][p].get("action") for p in range(2)]
            for t in range(len(steps) - 1)]

    rec = {
        "episode_id": info.get("EpisodeId"),
        "seed": info.get("seed"),
        "teams": info.get("TeamNames"),
        "rewards": d.get("rewards"),
        "statuses": d.get("statuses"),
        "module_version": d.get("module_version"),
        "n_steps": len(acts),
        "actions": acts,
    }

    field, market = [], []
    for p in range(2):
        field.append(_digest([[a[p].get("farmer"), a[p].get("hands")] if a[p] else None for a in acts]))
        market.append(_digest([a[p].get("market") if a[p] else None for a in acts]))
    rec["field_hash"], rec["market_hash"] = field, market

    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / f"{rec['episode_id']}.json.gz", "wt") as f:
        json.dump(rec, f, separators=(",", ":"))

    return {k: v for k, v in rec.items() if k != "actions"}


def main(zip_path):
    zip_path = str(Path(zip_path).resolve())
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
    print(f"{len(names)} episodes in {Path(zip_path).name}", flush=True)

    rows, errors = [], 0
    with ProcessPoolExecutor() as ex:
        for i, rec in enumerate(ex.map(one, [(zip_path, n) for n in names], chunksize=4), 1):
            if "error" in rec:
                errors += 1
            else:
                rows.append(rec)
            if i % 50 == 0:
                print(f"  {i}/{len(names)}", flush=True)

    # Путь наружу: индекс пула собирается из одного дампа, а для выбора
    # учителя надо сводить несколько дней, не затирая рабочий.
    index = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "corpus" / "index.csv"
    with open(index, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "seed", "team0", "team1", "reward0", "reward1",
                    "status0", "status1", "module_version", "n_steps",
                    "field_hash0", "field_hash1", "market_hash0", "market_hash1"])
        for r in sorted(rows, key=lambda r: r["episode_id"] or 0):
            t = r.get("teams") or [None, None]
            rw = r.get("rewards") or [None, None]
            st = r.get("statuses") or [None, None]
            w.writerow([r["episode_id"], r["seed"], t[0], t[1], rw[0], rw[1], st[0], st[1],
                        r["module_version"], r["n_steps"],
                        r["field_hash"][0], r["field_hash"][1],
                        r["market_hash"][0], r["market_hash"][1]])
    print(f"wrote {len(rows)} records -> {index}   ({errors} errors)")


if __name__ == "__main__":
    main(sys.argv[1])
