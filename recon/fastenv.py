"""Bit-exact drop-in for `env.run()` that skips the kaggle-environments wrapper.

Measured 2026-08-30 on kaggle-environments 1.32.7:
    env.run()          661 ms / episode   (1.5 eps/s/core)
    fast_run()        13.8 ms / episode   ( 72 eps/s/core)   -> x48

Verified bit-exact against `env.run()` on 20/20 seeds using the tier-9 reference
agent `closer_cleo` -- a real competitive agent, deliberately not `starter`.
Re-run `check_equivalence()` after every kaggle-environments upgrade: the whole
point of the harness is that it is the same engine, so any divergence means the
harness is wrong, not the engine.

Where the 54x comes from (cProfile, 5 episodes):
    structify()                4.33 s
    copy.deepcopy()            4.00 s
    jsonschema validation      2.48 s
    actual game logic          ~0.2 s
The interpreter itself was never the bottleneck.
"""

import types

from kaggle_environments.envs.kaggriculture import kaggriculture as K
from kaggle_environments.utils import structify

DEFAULTS = dict(
    episodeSteps=720, boardSize=10, startingMoney=3000, maxMarketOrdersPerTurn=10,
    turnsPerDay=24, shedCapacity=100, weedSpawnChance=0.005, townShopUnlockInterval=3,
    townShopSellInterval=4, townCenterSellInterval=24, farmHandCostMult=1,
)


class _Cfg(dict):
    """The interpreter reads config by attribute and by key; support both."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def new_episode(seed, **overrides):
    """Fresh initialized (env, state). `overrides` accepts any config key,
    including `marketParams` for domain randomisation of the price curves."""
    cfg = _Cfg(DEFAULTS, seed=seed, **overrides)
    env = types.SimpleNamespace(configuration=cfg, info={}, done=False)
    state = [
        structify({
            "action": {}, "reward": 0, "info": {}, "status": "ACTIVE",
            "observation": {"step": 0, "remainingOverageTime": 12},
        })
        for _ in range(2)
    ]
    K.interpreter(state, env)  # initialize
    return env, state


def observation_for(state, player, step):
    """The observation an agent sees. `step` is injected for both seats --
    the stock env only stamps it on seat 0, which is the known seat-1 bug."""
    obs = dict(state[0].observation)
    obs["player"] = player
    obs["step"] = step
    obs["private"] = state[player].observation.private
    return obs


def fast_run(agents, seed, steps=720, **overrides):
    """Play one episode. Returns [bank_p0, bank_p1].

    NOTE the `steps - 1`. The interpreter flips both seats to DONE at
    `step >= episodeSteps - 2`, so the last action index the framework ever
    executes is 718, and agents are called 719 times, not 720. Looping the full
    720 gives one extra turn of play and banks that are ~1% too high -- which is
    exactly the bug this harness shipped with until the cleo cross-check caught it.
    """
    env, state = new_episode(seed, **overrides)
    for step in range(steps - 1):
        for i, st in enumerate(state):
            st.action = agents[i](observation_for(state, i, step))
        state[0].observation.step = step
        K.interpreter(state, env)
    farms = state[0].observation.farms
    return [float(farms[i]["money"]) for i in range(2)]


def paired(agent_a, agent_b, seeds, **overrides):
    """CRN: same seed, both seatings. Returns (wins_a, wins_b, ties, mean_margin_a)."""
    wa = wb = t = 0
    margin = 0.0
    for s in seeds:
        for flip in (False, True):
            pair = (agent_b, agent_a) if flip else (agent_a, agent_b)
            r = fast_run(list(pair), s, **overrides)
            a, b = (r[1], r[0]) if flip else (r[0], r[1])
            margin += a - b
            if a > b:
                wa += 1
            elif b > a:
                wb += 1
            else:
                t += 1
    n = 2 * len(seeds)
    return wa, wb, t, margin / n


def load_agent(path):
    """Load an agent .py the way we use it for training/eval. Note this is NOT
    identical to how kaggle-environments loads a submitted file (it takes the last
    newly bound callable), so anything destined for submission must also be smoke
    tested through the real file-path interface."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("agentmod_" + path.replace("/", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("agent", "kaggle_submission_agent", "main"):
        if hasattr(mod, name):
            return getattr(mod, name)
    local = [v for v in vars(mod).values()
             if callable(v) and getattr(v, "__module__", "") == mod.__name__]
    return local[-1]


def check_equivalence(n_seeds=20, agent_path="refagents/closer_cleo.py"):
    """Positive control for the harness itself -- run before trusting ANY result.

    Deliberately checked against a real competitive agent, not `starter`. The
    720-vs-719 step bug was invisible on `starter` (30/30 "bit-exact") and showed
    up immediately on cleo: a control that cannot fail proves nothing.
    See HANDOFF_SIM_COMPS.md section 9b.
    """
    from kaggle_environments import make

    fn = load_agent(agent_path)
    bad = []
    for seed in range(n_seeds):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.run([agent_path, "pass"])
        ref = [env.steps[-1][i].reward for i in range(2)]
        got = fast_run([fn, K.pass_agent], seed)
        if ref != got:
            bad.append((seed, ref, got))
    print(f"{agent_path}: bit-exact on {n_seeds - len(bad)}/{n_seeds} seeds")
    for row in bad[:5]:
        print("  MISMATCH", row)
    return not bad


if __name__ == "__main__":
    import time

    check_equivalence()
    t = time.time()
    for s in range(20):
        fast_run([K.pass_agent, K.pass_agent], s)
    d = time.time() - t
    print(f"{d / 20 * 1000:.1f} ms/episode -> {20 / d:.0f} eps/s/core")
