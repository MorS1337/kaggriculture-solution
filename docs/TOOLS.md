# Tools

The final agent is `main.py` at the repository root. The scripts below are the research
instruments used to choose and validate the layers. They are included as methodology, not as
a turnkey benchmark: some of them expect `corpus/` and `recon/pubagents/`, which are campaign
data and are not redistributed.

## Head-to-head

```bash
# Paired seeds, both seats, bank gap. `random` and `starter` are built-in agents.
python vs_agent.py --agent main.py --opp random --games 8 --workers 4
```

`vs_agent.py` uses `recon/fastenv.py`, a bit-exact drop-in for `env.run()` that skips the
`kaggle-environments` wrapper (measured ~48x faster). The fast path is verified against the
real engine on reference seeds; re-run its equivalence check after every `kaggle-environments`
upgrade.

## Pool evaluation

```bash
python eval_agent.py --agent main.py --games 400 --workers 7
```

`eval_agent.py` plays against the tape pool built by `recon/pool.py`. To rebuild the pool:

```bash
# 1. Download a public episode dump (date must be yesterday or older).
kaggle datasets download -d kaggle/kaggriculture-episodes-YYYY-MM-DD --unzip -d /tmp/dump

# 2. Extract actions and the index (~20 GB -> ~40 MB).
python recon/extract_corpus.py /tmp/dump/<file>.zip

# 3. Fetch current public agents (Apache-2.0; kept out of git).
python recon/fetch_pubagents.py

# 4. Build the pool.
python recon/pool.py
```

## Diagnosis

| Tool | Question |
|---|---|
| `divergence.py` | Where exactly does our game diverge from a teacher/reference? |
| `plan_by_seed.py` | Which action plan is selected on a given seed? |
| `opening_sweep.py` | How sensitive is the result to the opening sequence? |
| `broad_opening.py` | Wider opening sweep across seeds. |
| `episode_spread.py` | Do outcomes diverge across episode samples? |

## Acceptance

```bash
tar -czf submission.tar.gz main.py
python accept_submission.py submission.tar.gz --opponent starter
```

This unpacks the archive, loads it with the official Kaggle loader, and plays a full episode.
It catches two silent packaging errors: no `__file__` at module scope, and the wrong last
callable becoming the entry point.

## Rule of thumb

A layer is promoted only if:

- paired seeds and both seats;
- two disjoint seed blocks;
- a fresh opponent pool (2-3 days old at most);
- the effect is larger than the block-to-block spread;
- the packaged artifact passes the official-loader acceptance test.
