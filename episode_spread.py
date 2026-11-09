"""Расходятся ли ИСХОДЫ эпизодов при семплировании.

Весь спор о токенах и температурах не имеет значения, если сыгранная сто раз
партия даёт один и тот же банк: градиенту политики не на чем работать. Меряем
то, что нужно RL, а не то, что удобно померить.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent) + "")
sys.path.insert(0, str(Path(__file__).resolve().parent) + "/weights/agents/bc-wide2")


def one(task):
    seed, temp, rep = task
    import numpy as np, torch
    torch.set_num_threads(1)
    from model.kaggriculture_bc.entity_train import load_policy
    from model.kaggriculture_bc.entity_data import encode_observation, decode_action
    import model.kaggriculture_bc.entity_policy as EP
    from rl.sampling import install
    from vs_agent import _load
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from recon.fastenv import new_episode, observation_for

    model, _ = load_policy(
        str(Path(__file__).resolve().parent) + "/weights/agents/bc-wide2/weights.pt",
        torch.device("cpu"))
    install(type(model), len(EP.ARGUMENTS))
    model._sample_temp = temp
    torch.manual_seed(10_000 + rep)
    opp = _load(str(Path(__file__).resolve().parent) + "/recon/pubagents/yhay81_shop-router-0909",
                None, f"o{rep}")
    env, state = new_episode(seed)
    for step in range(719):
        obs = observation_for(state, 0, step)
        f = encode_observation(obs)
        b = {k: torch.from_numpy(v).unsqueeze(0) for k, v in f.items()}
        with torch.inference_mode():
            a = model.generate(b)[0]
        state[0].action = decode_action(a.cpu().tolist(), 1 + len(obs["farms"][0]["hands"]))
        state[1].action = opp(observation_for(state, 1, step))
        state[0].observation.step = step
        K.interpreter(state, env)
    farms = state[0].observation.farms
    return temp, rep, float(farms[0]["money"]) - float(farms[1]["money"])


def main():
    import statistics as st
    seed = 990000
    tasks = [(seed, 0.0, 0)]
    for t in (1.0, 2.0):
        tasks += [(seed, t, r) for r in range(6)]
    with Pool(7) as p:
        rows = p.map(one, tasks)
    by = {}
    for temp, rep, m in rows:
        by.setdefault(temp, []).append(m)
    for temp in sorted(by):
        v = by[temp]
        spread = (max(v) - min(v)) if len(v) > 1 else 0.0
        uniq = len(set(round(x, 2) for x in v))
        print(f"темп {temp:<5} прогонов {len(v):>2}  различных исходов {uniq:>2}  "
              f"разброс {spread:>10,.0f}  разрывы: " + ", ".join(f"{x:,.0f}" for x in v))


if __name__ == "__main__":
    main()
