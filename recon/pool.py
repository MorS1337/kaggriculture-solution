#!/usr/bin/env python3
"""Turn corpus clusters into replayable opponents, and run a league.

Every strong ladder agent is an open-loop tape (0/719 field actions change with
the opponent, measured in recon/openloop.py), so replaying a recorded action
stream reproduces that team's strategy closely even on a fresh seed. That gives
us a pool of *current* opponents instead of the 17 Aug reference agents.

Two things this must not gloss over:
  * A tape recorded on one seed meets different weeds and shops on another, so
    it plays worse than it did in its own game. `python recon/pool.py --calib`
    measures that drop instead of assuming it away.
  * The popular losing cluster belongs in the pool too. It is what a large share
    of the ladder actually plays, and the metric that matters is win rate against
    the field as it is, not against the field's best members.
"""

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from recon.fastenv import fast_run, load_agent  # noqa: E402

DEFAULT = {"farmer": ["PASS"], "hands": [], "market": []}
_tapes = {}


def _load_tape(episode_id, seat):
    key = (episode_id, seat)
    if key not in _tapes:
        with gzip.open(REPO / "corpus" / "actions" / f"{episode_id}.json.gz", "rt") as f:
            rec = json.load(f)
        _tapes[key] = [t[seat] if isinstance(t[seat], dict) else DEFAULT
                       for t in rec["actions"]]
    return _tapes[key]


def tape_agent(episode_id, seat):
    """Replay a recorded stream. Indexed by day*24+hour, which is correct in both
    seats -- obs['step'] is not populated for seat 1 in serialized states."""
    def fn(obs):
        tape = _load_tape(episode_id, seat)
        t = obs["day"] * 24 + obs["hour"]
        return tape[t] if t < len(tape) else DEFAULT
    return fn


def clusters(min_games=6):
    with open(REPO / "corpus" / "index.csv") as f:
        rows = list(csv.DictReader(f))
    agg = defaultdict(lambda: {"g": 0, "w": 0, "t": 0, "teams": set(), "ex": None})
    for r in rows:
        for p in (0, 1):
            try:
                mine, theirs = float(r[f"reward{p}"]), float(r[f"reward{1-p}"])
            except (TypeError, ValueError):
                continue
            c = agg[r[f"field_hash{p}"]]
            c["g"] += 1
            c["teams"].add(r[f"team{p}"])
            if mine > theirs:
                c["w"] += 1
            elif mine == theirs:
                c["t"] += 1
            if c["ex"] is None:
                c["ex"] = (int(r["episode_id"]), p, int(r["seed"]), float(r[f"reward{p}"]))
    out = []
    for h, c in agg.items():
        if c["g"] >= min_games:
            c["wr"] = 100.0 * (c["w"] + 0.5 * c["t"]) / c["g"]
            out.append((h, c))
    return sorted(out, key=lambda kv: -kv[1]["wr"])


# Опубликованные агенты, живущие в `recon/pubagents/`. Каталог в gitignore --
# это чужой код под Apache-2.0, он воспроизводится из Kaggle скриптом
# `recon/fetch_pubagents.py`, а не хранится у нас.
#
# 2026-09-09: shop-router обыграл ВЕСЬ прежний пул со счётом 100% (20 пар из 20,
# оба места, +20 346 +/-3 018). Пул из лент 29 августа -- строго более слабый
# ярус, и принимать по нему решения больше нельзя: он сохраняет порядок, но
# сжимает разницу в разы (entity BC давал на нём +13 779 при 74%, а лоб в лоб
# тому же shop-router проигрывает 4 169).
PUBLIC = {
    "pub_shop_router": "yhay81_shop-router-0909",
    "pub_reactive_router": "aurax7_kaggriculture-reactive-router",
    "pub_state_router": "thomastschinkel_kaggriculture-public-state-router-74-5-win-rate",
}


def public_agents():
    """Те из опубликованных агентов, что реально лежат на диске."""
    out = {}
    for name, d in PUBLIC.items():
        p = REPO / "recon" / "pubagents" / d / "main.py"
        if p.exists():
            out[name] = ("ref", str(p), None, None, None)
    return out


def build_pool(n_strong=6, include_popular=True, include_public=True,
               include_tapes=True, include_refs=True):
    """Соперники по именам: живое поле + при необходимости прежние ярусы.

    `include_public` -- опубликованные агенты сентября, сегодняшнее золото.
    `include_tapes` / `include_refs` -- ленты 29 августа и эталоны 17 августа;
    оба яруса заведомо слабее и годятся только для отладки и сопоставимости со
    старыми числами.
    """
    pool = {}
    if include_public:
        pool.update(public_agents())
    if include_tapes:
        cl = clusters()
        for h, c in cl[:n_strong]:
            ep, seat, seed, bank = c["ex"]
            pool[f"tape_{h[:6]}_wr{c['wr']:.0f}"] = ("tape", ep, seat, seed, bank)
        if include_popular:
            h, c = sorted(cl, key=lambda kv: -kv[1]["g"])[0]
            ep, seat, seed, bank = c["ex"]
            pool[f"tape_{h[:6]}_popular_wr{c['wr']:.0f}"] = ("tape", ep, seat, seed, bank)
    if include_refs:
        for name in ("closer_cleo", "broker_bea", "slotter_silas"):
            pool[f"ref_{name}"] = ("ref", str(REPO / "refagents" / f"{name}.py"),
                                   None, None, None)
    return pool


def _mk(spec):
    kind = spec[0]
    return tape_agent(spec[1], spec[2]) if kind == "tape" else load_agent(spec[1])


def _duel(args):
    na, nb, sa, sb, seed = args
    a, b = _mk(sa), _mk(sb)
    w = l = t = 0
    margin = 0.0
    for flip in (False, True):
        pair = [b, a] if flip else [a, b]
        r = fast_run(pair, seed)
        ba, bb = (r[1], r[0]) if flip else (r[0], r[1])
        margin += ba - bb
        if ba > bb:
            w += 1
        elif bb > ba:
            l += 1
        else:
            t += 1
    return na, nb, w, l, t, margin / 2


def league(seeds):
    pool = build_pool()
    names = list(pool)
    jobs = [(x, y, pool[x], pool[y], s) for x, y in combinations(names, 2) for s in seeds]
    sc = defaultdict(lambda: {"w": 0, "l": 0, "t": 0, "n": 0, "m": 0.0})
    h2h = defaultdict(lambda: [0, 0, 0])
    with ProcessPoolExecutor() as ex:
        for na, nb, w, l, t, m in ex.map(_duel, jobs, chunksize=4):
            for n, ww, ll, mm in ((na, w, l, m), (nb, l, w, -m)):
                s = sc[n]
                s["w"] += ww; s["l"] += ll; s["t"] += t; s["n"] += 2; s["m"] += mm
            h2h[(na, nb)][0] += w; h2h[(na, nb)][1] += l; h2h[(na, nb)][2] += t

    print(f"\nЛига: {len(names)} участников, {len(jobs) * 2} игр, "
          f"{len(seeds)} сидов, оба сиденья\n")
    print(f"{'агент':34s} {'W-L-T':>12s} {'win%':>7s} {'ср. маржа':>12s}")
    print("-" * 70)
    for n in sorted(names, key=lambda n: -(sc[n]["w"] + 0.5 * sc[n]["t"]) / max(1, sc[n]["n"])):
        s = sc[n]
        wr = 100.0 * (s["w"] + 0.5 * s["t"]) / s["n"]
        print(f"{n:34s} {s['w']:3d}-{s['l']:3d}-{s['t']:<3d} {wr:6.1f}% "
              f"{s['m'] / (s['n'] / 2):12,.0f}")
    return sc


def calibrate():
    """How much does a tape lose when replayed on seeds it was not recorded on?"""
    pool = {k: v for k, v in build_pool().items() if v[0] == "tape"}
    print(f"\n{'лента':34s} {'банк в своей партии':>20s} {'на своём сиде':>15s} "
          f"{'на 8 чужих':>12s} {'потеря':>9s}")
    print("-" * 96)
    for name, spec in pool.items():
        _, ep, seat, seed, bank = spec
        a = _mk(spec)
        own = fast_run([a, a], seed)[0]
        fresh = [fast_run([a, a], s)[0] for s in range(500, 508)]
        mean = sum(fresh) / len(fresh)
        drop = 100.0 * (mean - own) / own if own else 0.0
        print(f"{name:34s} {bank:20,.0f} {own:15,.0f} {mean:12,.0f} {drop:8.1f}%")
    print("\n«на своём сиде» = лента против самой себя на записанном сиде; "
          "разница с первым столбцом = вклад реального соперника.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start", type=int, default=300, help="first seed; use a fresh range to replicate")
    a = ap.parse_args()
    if a.calib:
        calibrate()
    else:
        league(list(range(a.start, a.start + a.seeds)))
