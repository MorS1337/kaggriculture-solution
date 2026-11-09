# Attribution / NOTICE

This repository is a competition artifact for **Kaggle Kaggriculture** (2026).

## Game and environment

- Kaggriculture game rules, engine, and `kaggle-environments` are the property of their
  respective authors and the competition organizers.
- The rules reference is in [`docs/GAME_RULES.md`](docs/GAME_RULES.md).
- The competition header art in `assets/kaggriculture-header.png` is the official Kaggle
  competition art, used for identification only. Copyright belongs to Kaggle / the
  competition organizers.

## Final agent: public Apache-2.0 chassis + team layers

The final `main.py` is a derivative of the public Kaggle notebook:

- **Ahmed Berat Özer**, *Kaggriculture V57 — Funding-Order Invariant*, Apache-2.0.

Its upstream lineage, as retained in the source notices, includes:

- **Nathan Jacob** — Pipe16 HybridOpening;
- **Thomas Tschinkel** — Metav4 production/market controller;
- **Dmitrii Gluzdov** — physical terminal rescue;
- **yhay81** — shop-router / action tapes;
- **prvsiyan** — *The Moon Counts Melons* (ADV/VQ/FRO layers);
- **haideptry** — *The Shepherd's Ledger: Herd-Safe Sovereign* (HR/HP layers);
- **tetsutani** — DPTS market-execution layers;
- **aurax7**, **destbreso**, and other public authors whose notices are retained in
  `main.py`.

The Apache-2.0 license text and all upstream notices are retained in `main.py` and
[`LICENSE`](LICENSE). The authors of the public notebooks retain all rights to their code
and ideas.

## Team changes ("loss volos" / "MISIS loss volos")

The team's layers on top of the public chassis include:

- enabling `clamp_sells` and `room_guard`;
- route remaps, weed fix, wheat-seed cap, race 45, carrot -15, V92_P frontier 1.5;
- ADV/VQ/FRO and HR/HP ports validated on the live pool;
- starvation rescue and day-0 cash reserve;
- rival-lead learner and sale-timing layers;
- HP forecast window 8 and PET_CAFE carrot tilt;
- opening market sequence (EXP791/792);
- shadow wheat/fertilizer round trips (EXP793/794);
- order reorder (EXP796);
- final packaging and acceptance harness.

These changes are documented in the commit history of the research repository and in the
comments of `main.py`. The public chassis is **not** claimed as original work by this team.

## No relicensing of third-party code

This repository does not relicense third-party code. The presence of derived Apache-2.0
components here is for reproducibility of a Kaggle competition result. For any commercial or
redistributed use, check the original notebooks, the Apache-2.0 license, and Kaggle's terms
first.
