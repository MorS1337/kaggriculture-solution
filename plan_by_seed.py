#!/usr/bin/env python3
"""Какой из тринадцати планов ленты выбирается на каждом посеве.

Зачем. E92 оставил открытым вопрос: почему на 9 партиях из 50 клон с первых
ходов оказывается на траектории, которой нет в корпусе, если на остальных 41 он
повторяет ленту точно. Слово «покрытие» станет проверяемым, если окажется, что
развальные посевы сидят на плане, которого в обучающем корпусе почти нет.

Как лента выбирает план: на ходу 144 (`ROUTE_STEP`) она смотрит первые две
открытые лавки и берёт `SHOP_PLANS[(лавка0, лавка1)]`, по умолчанию 0. На ходу
648 все переходят на план 2. То есть план -- функция посева, и его можно снять,
доиграв всего 145 ходов вместо 719.

ВАЖНО про запуск: этот скрипт нельзя выполнять через `python3 - <<EOF`. На маке
`multiprocessing` стартует рабочих методом *spawn*, каждый заново импортирует
`__main__` по пути файла, а у стандартного ввода пути нет -- рабочие падают с
`FileNotFoundError: '<stdin>'`, Pool поднимает новых, и задача крутится вечно,
не посчитав ничего. Один раз это стоило 82 минут и шести ядер.

    python3 plan_by_seed.py --group замер 990000 50
"""
import argparse
import importlib.util
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

_CACHE = {}


def _teacher():
    """Лента поднимается ОДИН раз на рабочего: в actions.json пять мегабайт,
    и разбор их на каждый посев был главной статьёй расхода времени."""
    if "mod" not in _CACHE:
        spec = importlib.util.spec_from_file_location(
            "plan_teacher", REPO / "build/market_agent/main.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["plan_teacher"] = m
        spec.loader.exec_module(m)
        _CACHE["mod"] = m
        from vs_agent import _load
        _CACHE["me"] = _load(str(REPO / "build/market_agent"), None, "plan_me")
        _CACHE["opp"] = _load(
            str(REPO / "recon/pubagents/yhay81_shop-router-0909"), None, "plan_opp")
    return _CACHE


def plan_of(seed):
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from recon.fastenv import new_episode, observation_for
    c = _teacher()
    env, state = new_episode(seed)
    for step in range(145):
        obs = observation_for(state, 0, step)
        if step == 144:
            shops = tuple(obs["town"]["unlocked_shops"][:2])
            return seed, shops, c["mod"].SHOP_PLANS.get(shops, 0)
        state[0].action = c["me"](obs)
        state[1].action = c["opp"](observation_for(state, 1, step))
        state[0].observation.step = step
        K.interpreter(state, env)
    return seed, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs=3, action="append", metavar=("ИМЯ", "ПОСЕВ0", "СКОЛЬКО"),
                    required=True, help="имя группы, первый посев, число посевов")
    ap.add_argument("--step", type=int, default=1, help="шаг по посевам внутри группы")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mark", type=int, nargs="*", default=[],
                    help="посевы, которые выделить в конце отдельным списком")
    a = ap.parse_args()

    groups = {name: list(range(int(s0), int(s0) + int(n) * a.step, a.step))
              for name, s0, n in a.group}
    out = {}
    with Pool(a.workers) as p:
        for name, seeds in groups.items():
            out[name] = p.map(plan_of, seeds)

    for name, rows in out.items():
        c = Counter(r[2] for r in rows)
        tot = sum(c.values())
        print(f"\n{name}: {tot} посевов")
        for plan, n in sorted(c.items()):
            print(f"  план {plan:2d}: {n:3d}  ({100 * n / tot:4.1f}%)")

    if a.mark:
        print("\nвыделенные посевы:")
        for rows in out.values():
            for seed, shops, plan in rows:
                if seed in a.mark:
                    print(f"  {seed}  план {plan:2d}  лавки {shops}")


if __name__ == "__main__":
    main()
