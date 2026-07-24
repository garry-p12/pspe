CONDA_ENV ?= pspe
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo /opt/anaconda3)
PY := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python
PIP := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/pip

.PHONY: setup setup-llm test test-all canary validate-baselines smoke data \
        train-simulate train-plan train-perceive train-explain train-e2e rate \
        baselines-simulate baselines-plan ablations results clean tb

## ---- Phase 0: environment -------------------------------------------------
setup:
	conda create -y -n $(CONDA_ENV) python=3.11 || true
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup-llm:
	$(PIP) install -e ".[llm]"

smoke:
	$(PY) scripts/check_env.py

# Default suite excludes the slow safe-RL convergence checks and the canary.
test:
	$(PY) -m pytest tests -q -m "not slow and not canary"

test-all:
	$(PY) -m pytest tests -q -m "not canary"

# External-benchmark correctness check for the FNO. Run once, not per-commit.
canary:
	$(PY) -m pytest tests -q -m canary -s

# Safe-RL baselines against a standard benchmark. Needs safety-gymnasium,
# which does not build on macOS/arm64 - run on Linux or Colab.
validate-baselines:
	$(PY) baselines/validate_safety_gym.py

## ---- Phase 1: Simulate ----------------------------------------------------
data:
	$(PY) scripts/generate_data.py testbed=dar
	$(PY) scripts/generate_data.py testbed=swe
	$(PY) scripts/generate_data.py testbed=rdf

train-simulate:
	$(PY) scripts/train_simulate.py testbed=dar

baselines-simulate:
	$(PY) baselines/run_operator_baselines.py testbed=dar

## ---- Phase 2: Plan --------------------------------------------------------
train-plan:
	$(PY) scripts/train_plan.py testbed=dar

baselines-plan:
	$(PY) baselines/run_safe_rl_baselines.py testbed=dar

## ---- Phase 3: Perceive ----------------------------------------------------
train-perceive:
	$(PY) scripts/train_perceive.py

## ---- Phase 4: Explain -----------------------------------------------------
train-explain:
	$(PY) scripts/train_explain.py

rate:
	$(PY) eval/human_rating.py --briefs runs/explain/briefs.jsonl

## ---- Phase 5: Integration -------------------------------------------------
train-e2e:
	$(PY) scripts/train_e2e.py

ablations:
	$(PY) eval/run_ablations.py

results:
	$(PY) eval/report.py

tb:
	$(CONDA_BASE)/envs/$(CONDA_ENV)/bin/tensorboard --logdir runs

clean:
	rm -rf runs/* data/pde_testbeds/*.npz outputs/ .pytest_cache
