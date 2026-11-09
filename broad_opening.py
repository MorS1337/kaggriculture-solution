"""Есть ли открытие с ШИРОКИМ выигрышем, а не заточенное под один файл.

Отложенный блок 991000+ (подбор шёл на 990000+). Пять соперников, пятнадцать пар.
"""
import importlib.util, statistics as st, sys
from multiprocessing import Pool
R = str(__import__("pathlib").Path(__file__).resolve().parent)
sys.path.insert(0, R)

OPPS = {"СИЛЬНЫЙ": "recon/pubagents/reyhanksatria_kaggriculture-dynamic-route-agent",
        "shop-router": "recon/pubagents/yhay81_shop-router-0909",
        "aurax7": "recon/pubagents/aurax7_kaggriculture-reactive-router",
        "thomastsch": "recon/pubagents/thomastschinkel_kaggriculture-public-state-router-74-5-win-rate",
        "municef1": "recon/pubagents/municef1_lb-2448-single-file-kaggriculture-agent"}
VARIANTS = [None, (13,30,30), (5,15,15), (5,30,30), (13,15,15), (5,15,20), (3,30,30)]

def one(task):
    seed, flip, var, oname = task
    from vs_agent import _load
    from recon.fastenv import fast_run, new_episode, observation_for
    spec = importlib.util.spec_from_file_location("tp", f"{R}/build/market_agent/main.py")
    m = importlib.util.module_from_spec(spec); sys.modules["tp"] = m; spec.loader.exec_module(m)
    env, state = new_episode(seed)
    m.agent(observation_for(state, 0, 0))
    if var is not None:
        a, b, s = var
        op = [['BUY_PRODUCT','WHEAT',a],['BUY_PRODUCT','WHEAT',b],['SELL','WHEAT',s]]
        for t in m._POLICY.tapes:
            t[0] = dict(t[0], market=[list(o) for o in op])
    me = lambda o: m.agent(o)
    opp = _load(f"{R}/{OPPS[oname]}", None, f"o{seed}{flip}")
    pair = [opp, me] if flip else [me, opp]
    r = fast_run(pair, seed)
    x, y = (r[1], r[0]) if flip else (r[0], r[1])
    return var, oname, seed, x - y

def main():
    tasks = [(991000+i, flip, v, o) for o in OPPS for v in VARIANTS
             for i in range(15) for flip in (False, True)]
    with Pool(7) as p:
        rows = p.map(one, tasks)
    hdr = "открытие".ljust(16) + "".join(o.rjust(13) for o in OPPS) + "худшее".rjust(13)
    print(hdr); print("-" * len(hdr))
    for v in VARIANTS:
        name = "как есть" if v is None else f"{v[0]}+{v[1]}/{v[2]}"
        cells, vals = [], []
        for o in OPPS:
            byp = {}
            for vv, oo, seed, d in rows:
                if vv == v and oo == o: byp.setdefault(seed, []).append(d)
            mval = st.mean([st.mean(x) for x in byp.values()])
            vals.append(mval); cells.append(f"{mval:+,.0f}".rjust(13))
        print(name.ljust(16) + "".join(cells) + f"{min(vals):+,.0f}".rjust(13))
    print("\nширокий выигрыш = наибольшее ХУДШЕЕ значение по строке")

if __name__ == "__main__":
    main()
