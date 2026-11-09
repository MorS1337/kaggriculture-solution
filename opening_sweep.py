import importlib.util, statistics as st, sys
from multiprocessing import Pool
REPO = str(__import__("pathlib").Path(__file__).resolve().parent)
sys.path.insert(0, REPO)

VARIANTS = [None, (13,30,30), (5,15,15), (3,10,10), (8,20,20), (5,15,20)]
OPPS = {
    "shop-router":  "recon/pubagents/yhay81_shop-router-0909",
    "aurax7":       "recon/pubagents/aurax7_kaggriculture-reactive-router",
    "thomastschin": "recon/pubagents/thomastschinkel_kaggriculture-public-state-router-74-5-win-rate",
    "municef1":     "recon/pubagents/municef1_lb-2448-single-file-kaggriculture-agent",
    "СИЛЬНЫЙ":      "recon/pubagents/reyhanksatria_kaggriculture-dynamic-route-agent",
}

def one(task):
    seed, flip, var, oname = task
    from vs_agent import _load
    from recon.fastenv import fast_run, new_episode, observation_for
    spec = importlib.util.spec_from_file_location("tp", f"{REPO}/build/market_agent/main.py")
    m = importlib.util.module_from_spec(spec); sys.modules["tp"] = m; spec.loader.exec_module(m)
    env, state = new_episode(seed)
    m.agent(observation_for(state, 0, 0))
    if var is not None:
        b1, b2, s = var
        op = [['BUY_PRODUCT','WHEAT',b1],['BUY_PRODUCT','WHEAT',b2],['SELL','WHEAT',s]]
        for t in m._POLICY.tapes:
            t[0] = dict(t[0], market=[list(o) for o in op])
    me = lambda obs: m.agent(obs)
    opp = _load(f"{REPO}/{OPPS[oname]}", None, f"o{seed}{flip}")
    pair = [opp, me] if flip else [me, opp]
    r = fast_run(pair, seed)
    a, b = (r[1], r[0]) if flip else (r[0], r[1])
    return var, oname, seed, a - b

def main():
    tasks = [(990000+i, flip, v, o) for o in OPPS for v in VARIANTS
             for i in range(20) for flip in (False, True)]
    with Pool(7) as p:
        rows = p.map(one, tasks)
    hdr = "открытие".ljust(18) + "".join(o.rjust(16) for o in OPPS)
    print(hdr); print("-" * len(hdr))
    for v in VARIANTS:
        name = "как есть" if v is None else f"{v[0]}+{v[1]}/{v[2]}"
        cells = []
        for o in OPPS:
            byp = {}
            for vv, oo, seed, d in rows:
                if vv == v and oo == o: byp.setdefault(seed, []).append(d)
            pm = [st.mean(x) for x in byp.values()]
            cells.append(f"{st.mean(pm):+,.0f}".rjust(16))
        print(name.ljust(18) + "".join(cells))
    print("\nпогрешность по парам (20 пар на клетку) -- отдельной строкой для лучших:")
    for v in [(5,15,15), (13,30,30)]:
        for o in ("СИЛЬНЫЙ", "shop-router"):
            byp = {}
            for vv, oo, seed, d in rows:
                if vv == v and oo == o: byp.setdefault(seed, []).append(d)
            pm = [st.mean(x) for x in byp.values()]
            pw = [st.mean([1.0 if x > 0 else 0.0 for x in y]) for y in byp.values()]
            print(f"  {v[0]}+{v[1]}/{v[2]} против {o:<12} разрыв {st.mean(pm):+9,.0f} ± {st.stdev(pm)/len(pm)**0.5:,.0f}"
                  f"   побед {100*st.mean(pw):5.1f}% ± {100*st.stdev(pw)/len(pw)**0.5:.1f}")

if __name__ == "__main__":
    main()
