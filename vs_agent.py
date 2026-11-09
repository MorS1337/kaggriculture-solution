#!/usr/bin/env python3
"""Голова к голове: наш агент против конкретного соперника, оба места, парно.

`eval_agent.py` меряет против пула лент из дампа. Этот скрипт нужен для другого
вопроса -- как мы стоим против ЖИВОГО опубликованного агента, который сейчас
массово заливают на лидерборд. Соперник задаётся так же, как кандидат: путь к
main.py или к каталогу с ним.

    python3 vs_agent.py --agent policy/own_main.py \
        --opp recon/pubagents/yhay81_shop-router-0909 --games 40
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


def _load(spec, entrypoint=None, tag="a"):
    """main.py -> вызываемый агент. Модули грузятся под РАЗНЫМИ именами:
    оба агента живут в одном процессе, и общее имя в sys.modules стёрло бы один
    другим."""
    if spec in ("pass", "random", "starter"):
        from kaggle_environments import make
        return make("kaggriculture").agents[spec]
    p = Path(spec)
    if p.is_dir():
        p = p / "main.py"
    name = f"cand_{tag}"
    # Каталог агента -- в пути: многофайловые агенты импортируют соседей
    # (market_primitives, sheep_admission, upstream). Настоящий загрузчик
    # Kaggle распаковывает тарбол и работает из его каталога, поэтому там
    # такие импорты живут, а у нас падали с ModuleNotFoundError.
    d = str(p.parent.resolve())
    if d not in sys.path:
        sys.path.insert(0, d)
    mod_spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[name] = mod
    mod_spec.loader.exec_module(mod)
    for nm in ([entrypoint] if entrypoint else []) + ["agent", "act", "main"]:
        fn = getattr(mod, nm, None)
        if callable(fn):
            return _one_arg(fn)
    raise SystemExit(f"{p}: не нашёл вызываемое. Укажите --entrypoint.")


def _one_arg(fn):
    """`fast_run` зовёт агента одним аргументом, а часть чужих main.py требует
    и `configuration`. Пробуем один аргумент, при TypeError -- два. Пустой dict
    в роли конфигурации законен: правила эпизода в fastenv стандартные, а все
    известные потребители проверяют лишь отличия от значений по умолчанию."""
    def call(obs):
        try:
            return fn(obs)
        except TypeError:
            return fn(obs, {})
    return call


def duel(t):
    spec, entry, ospec, oentry, seed, flip = t
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from recon.fastenv import fast_run
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    me = _load(spec, entry, "me")
    opp = _load(ospec, oentry, "opp")
    pair = [opp, me] if flip else [me, opp]
    r = fast_run(pair, seed)
    us, them = (r[1], r[0]) if flip else (r[0], r[1])
    return {"seed": int(seed), "candidate_seat": 1 if flip else 0,
            "candidate_bank": float(us), "opponent_bank": float(them)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--entrypoint", default=None)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--opp-entrypoint", default=None)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=990000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--rows", type=Path, default=None)
    a = ap.parse_args()
    if a.games % 2:
        raise SystemExit("--games должно быть чётным: каждая пара -- два места")
    tasks = [(a.agent, a.entrypoint, a.opp, a.opp_entrypoint,
              a.seed0 + i // 2, i % 2 == 1) for i in range(a.games)]
    with Pool(a.workers) as p:
        rows = p.map(duel, tasks)
    if a.rows:
        a.rows.parent.mkdir(parents=True, exist_ok=True)
        a.rows.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    byp = {}
    for r in rows:
        byp.setdefault(r["seed"], []).append(r)
    pm = [st.mean([x["candidate_bank"] - x["opponent_bank"] for x in v])
          for v in byp.values()]
    pw = [st.mean([1.0 if x["candidate_bank"] > x["opponent_bank"] else 0.0
                   for x in v]) for v in byp.values()]
    n = len(pm)
    se_m = st.stdev(pm) / n ** 0.5 if n > 1 else 0.0
    se_w = st.stdev(pw) / n ** 0.5 if n > 1 else 0.0
    print(f"{a.agent}  ПРОТИВ  {a.opp}")
    print(f"партий {len(rows)}, пар {n}, оба места\n")
    print(f"  доля побед   {100*st.mean(pw):6.2f}% ± {100*se_w:.2f}")
    print(f"  наш банк     {st.mean([r['candidate_bank'] for r in rows]):10,.0f}")
    print(f"  соперник     {st.mean([r['opponent_bank'] for r in rows]):10,.0f}")
    print(f"  РАЗРЫВ       {st.mean(pm):10,.0f} ± {se_m:,.0f}")
    for seat in (0, 1):
        d = [r["candidate_bank"] - r["opponent_bank"]
             for r in rows if r["candidate_seat"] == seat]
        if d:
            print(f"  место {seat}      {st.mean(d):10,.0f}  "
                  f"побед {100*sum(x>0 for x in d)/len(d):.0f}%")


if __name__ == "__main__":
    main()
