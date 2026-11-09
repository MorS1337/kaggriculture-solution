#!/usr/bin/env python3
"""Любой агент против живого пула: парно, оба места ПРОТИВ КАЖДОГО соперника.

Принимает то же, что упряжка Codex: путь к main.py, каталог с main.py, или
встроенные `pass`/`random`/`starter`. Соперник -- шесть сильнейших политик из
свежего дампа Kaggle плюс три реактивных эталона, а не встроенный `starter`,
который за сезон зарабатывает 551 монету (см. POOL.md).

Раздача мест (исправлена по разбору codex, exchange/0004): пара `i//2` задаёт
и соперника, и сид, а чётность `i` -- место. Прежняя версия брала соперника как
`i % n_opp`, из-за чего НИ ОДНА из двухсот пар не играла одного соперника, а
каждый соперник видел только одно наше место: таблица по соперникам была
смешана с эффектом места.

    python3 eval_agent.py --agent /path/to/main.py --games 400 --workers 28
"""
import argparse
import importlib.util
import json
import random
import statistics as st
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
_W = {}


def _load(spec, entrypoint=None):
    """main.py -> вызываемый агент: файл, каталог, или встроенный."""
    if spec in ("pass", "random", "starter"):
        from kaggle_environments import make
        return make("kaggriculture").agents[spec]
    p = Path(spec)
    if p.is_dir():
        p = p / "main.py"
    mod_spec = importlib.util.spec_from_file_location("cand_agent", p)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules["cand_agent"] = mod
    mod_spec.loader.exec_module(mod)
    names = ([entrypoint] if entrypoint else []) + ["agent", "act", "main"]
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise SystemExit(f"{p}: не нашёл вызываемое из {names}. "
                     "Укажите --entrypoint.")


def duel(t):
    spec, entry, oi, seed, flip = t
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from recon.fastenv import fast_run
    from recon.pool import _mk, build_pool
    # Состояние агента между партиями не переносится: объект строится заново,
    # генераторы сбрасываются от сида партии. Иначе порядок задач и число
    # рабочих влияют на результат, и прогон перестаёт быть воспроизводимым для
    # агентов с внутренней памятью или выборкой.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if "opps" not in _W:
        pool = build_pool(n_strong=6)
        _W["names"] = list(pool)
        _W["specs"] = list(pool.values())
    opp = _mk(_W["specs"][oi])
    me = _load(spec, entry)
    pair = [opp, me] if flip else [me, opp]
    r = fast_run(pair, seed)
    us, them = (r[1], r[0]) if flip else (r[0], r[1])
    return {"opponent": _W["names"][oi], "seed": int(seed),
            "candidate_seat": 1 if flip else 0,
            "candidate_bank": float(us), "opponent_bank": float(them)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--entrypoint", default=None,
                    help="имя вызываемого в main.py, если не agent/act/main")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=990000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--rows", type=Path, default=None,
                    help="куда сложить построчные результаты (JSON)")
    a = ap.parse_args()
    from recon.pool import build_pool
    n_opp = len(build_pool(n_strong=6))
    if a.games % 2:
        raise SystemExit("--games должно быть чётным: каждая пара -- два места")
    n_pairs = a.games // 2
    if n_pairs % n_opp:
        print(f"ВНИМАНИЕ: {n_pairs} пар не кратно {n_opp} соперникам — "
              f"покрытие неравномерное", file=sys.stderr)
    tasks = []
    for i in range(a.games):
        pair = i // 2
        tasks.append((a.agent, a.entrypoint, pair % n_opp,
                      a.seed0 + pair, i % 2 == 1))
    with Pool(a.workers) as p:
        rows = p.map(duel, tasks)
    if a.rows:
        a.rows.parent.mkdir(parents=True, exist_ok=True)
        a.rows.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    # Погрешность считается по ПАРАМ, а не по партиям: две партии одной пары
    # играются на одном сиде против одного соперника и независимыми не являются.
    byp = {}
    for r in rows:
        byp.setdefault((r["opponent"], r["seed"]), []).append(r)
    pair_margin, pair_win = [], []
    for k, v in byp.items():
        m = [x["candidate_bank"] - x["opponent_bank"] for x in v]
        pair_margin.append(st.mean(m))
        pair_win.append(st.mean([1.0 if d > 0 else 0.0 for d in m]))
    n = len(pair_margin)
    se_m = st.stdev(pair_margin) / n ** 0.5 if n > 1 else 0.0
    se_w = st.stdev(pair_win) / n ** 0.5 if n > 1 else 0.0
    print(f"агент: {a.agent}")
    print(f"партий {len(rows)}, пар {n}, оба места против КАЖДОГО соперника, "
          f"пул из {n_opp}\n")
    print(f"  доля побед   {100*st.mean(pair_win):6.2f}% ± {100*se_w:.2f}")
    print(f"  наш банк     {st.mean([r['candidate_bank'] for r in rows]):10,.0f}")
    print(f"  соперник     {st.mean([r['opponent_bank'] for r in rows]):10,.0f}")
    print(f"  РАЗРЫВ       {st.mean(pair_margin):10,.0f} ± {se_m:,.0f}")
    agg = {}
    for r in rows:
        d = agg.setdefault(r["opponent"], {0: [], 1: []})
        d[r["candidate_seat"]].append(r["candidate_bank"] - r["opponent_bank"])
    print(f"\n{'соперник':<26}{'пар':>5}{'разрыв место 0':>16}{'место 1':>12}"
          f"{'побед':>8}")
    for nm, d in agg.items():
        both = d[0] + d[1]
        w = 100 * sum(x > 0 for x in both) / len(both)
        print(f"{nm:<26}{len(d[0]):5d}{st.mean(d[0]):16,.0f}"
              f"{st.mean(d[1]):12,.0f}{w:7.1f}%")


if __name__ == "__main__":
    main()
