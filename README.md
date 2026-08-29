# RIVF / GRAFT-v1

Canonical method: [`pipeline.md`](pipeline.md)  
Canonical experiments: [`experiment_v2.md`](experiment_v2.md)  
Canonical configs: [`configs/README.md`](configs/README.md)

## Setup and validation

```bash
python -m pip install -e '.[dev]'
python -m spacy download en_core_web_sm
pytest -q
python scripts/validate_benchmarks.py
python -m rivf.experiments.run_experiment configs/graft_v1_smoke.yaml --preflight
```

## Safe smoke test

The smoke config uses one development QID and does not call an LLM:

```bash
python -m rivf.experiments.run_experiment configs/graft_v1_smoke.yaml
```

Results under `results/` and `result_v0.1/` predate GRAFT-v1 unless explicitly
produced by a config listed in `configs/README.md`.
