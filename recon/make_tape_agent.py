#!/usr/bin/env python3
"""Generate a standalone, submittable agent that replays a recorded tape.

The corpus is not present on the Kaggle runner, so the action stream is embedded
in the file (zlib + base85, the same way the public reference agents do it).

Indexing uses day*24 + hour, never obs["step"]: the serialized observation for
seat 1 does not carry `step`, and an agent that keys off it replays turn 0
forever in half its games.

    python recon/make_tape_agent.py 102520529 0 --name hayashi_a
"""
import argparse, base64, gzip, json, sys, zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TEMPLATE = '''"""Replay of a public ladder tape (episode {ep}, seat {seat}, team: {team}).

Source: Kaggle's own published Daily Top Episodes dataset. Used as a strength
floor and as a pool opponent while our own agent is built -- not as our entry.
Recorded result in its own game: {bank:,.0f} vs {opp:,.0f}.
"""
import base64
import json
import zlib

_TAPE = json.loads(zlib.decompress(base64.b85decode(
    "{blob}"
)).decode())

_DEFAULT = {{"farmer": ["PASS"], "hands": [], "market": []}}


def agent(obs):
    # day*24+hour, not obs["step"]: seat 1 does not receive `step`.
    try:
        t = int(obs["day"]) * 24 + int(obs["hour"])
    except Exception:
        return _DEFAULT
    if 0 <= t < len(_TAPE):
        a = _TAPE[t]
        return a if isinstance(a, dict) else _DEFAULT
    return _DEFAULT
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=int)
    ap.add_argument("seat", type=int)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    with gzip.open(REPO / "corpus" / "actions" / f"{a.episode}.json.gz", "rt") as f:
        rec = json.load(f)
    tape = [t[a.seat] if isinstance(t[a.seat], dict) else None for t in rec["actions"]]
    blob = base64.b85encode(zlib.compress(json.dumps(
        tape, separators=(",", ":")).encode(), 9)).decode()

    src = TEMPLATE.format(ep=a.episode, seat=a.seat, blob=blob,
                          team=(rec.get("teams") or ["?", "?"])[a.seat],
                          bank=rec["rewards"][a.seat], opp=rec["rewards"][1 - a.seat])
    out = REPO / "policy" / f"{a.name}.py"
    out.write_text(src)
    print(f"{out.relative_to(REPO)}  {len(src):,} bytes  ({len(tape)} turns)")

    # self-check: the generated file must reproduce the recorded game exactly
    sys.path.insert(0, str(REPO))
    from recon.fastenv import fast_run, load_agent
    from recon.pool import tape_agent
    me = load_agent(str(out))
    opp = tape_agent(a.episode, 1 - a.seat)
    pair = [opp, me] if a.seat else [me, opp]
    r = fast_run(pair, rec["seed"])
    want = [float(x) for x in rec["rewards"]]
    print(f"  reproduces its own game: {r == want}  (got {r}, want {want})")
    return r == want


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
