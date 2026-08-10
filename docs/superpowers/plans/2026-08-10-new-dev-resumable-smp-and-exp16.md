# new_dev Resumable SMP and Exp16 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make future SMP checkpoints preserve DiffNormalizer state, then add reproducible Exp13/Exp14 and Exp16 training pipelines on `new_dev`.

**Architecture:** A project-local runner subclass persists the only long-lived SMP reward-calibration state. New task registrations use that runner. Two isolated shell pipelines prepare fully mirrored priors and launch only explicit experiment groups; Exp16 uses a fresh prior and from-scratch policies.

**Tech Stack:** Python 3.13, PyTorch, mjlab/rsl-rl, Bash, unittest, Ruff.

---

### Task 1: Serialize and validate DiffNormalizer state

**Files:**
- Modify: `src/smp/rl/utils.py`
- Create: `tests/test_smp_normalizer_resume.py`

- [x] **Step 1: Write failing tests**

Add tests that update a 4-timestep normalizer at two timesteps, round-trip its state through a second instance, and assert `mean`, `count`, and the following normalized value match. Add malformed-state cases for missing schema, wrong shape, non-finite mean, non-positive mean, and negative count.

- [x] **Step 2: Run red test**

Run: `./.venv/bin/python -m unittest tests/test_smp_normalizer_resume.py`

Expected: failure because `DiffNormalizer` has no serialization API.

- [x] **Step 3: Implement minimal state API**

Add `state_dict()` containing version `1`, `num_timesteps`, CPU-cloned `mean/count`, `min_value`, and `max_count`. Add `load_state_dict()` that validates every key, type, one-dimensional expected shape, finite/strictly-positive mean, non-negative count, matching timestep count, then copies tensors to the existing device and dtypes.

- [x] **Step 4: Run green test**

Run: `./.venv/bin/python -m unittest tests/test_smp_normalizer_resume.py`

Expected: all normalizer tests pass.

### Task 2: Persist normalizer state through a project runner

**Files:**
- Create: `src/smp/rl/runner.py`
- Modify: `src/smp/rl/__init__.py`
- Modify: `tests/test_smp_normalizer_resume.py`

- [x] **Step 1: Write failing runner tests**

Use a minimal fake env and monkeypatched parent save/load to assert `SmpOnPolicyRunner.save()` stores state at `infos['smp_normalizer']`, load restores it, and a strict legacy load raises `RuntimeError` when only `infos['env_state']` exists.

- [x] **Step 2: Run red test**

Run: `./.venv/bin/python -m unittest tests/test_smp_normalizer_resume.py`

Expected: import failure for `SmpOnPolicyRunner`.

- [x] **Step 3: Implement runner**

Subclass `MjlabOnPolicyRunner`. Merge any incoming infos with a top-level `smp_normalizer` state before calling parent save. After parent load, restore state when present. If missing, retain startup state and either warn (non-strict) or raise a clear strict-resume error. Export the class from `smp.rl`.

- [x] **Step 4: Run green tests**

Run: `./.venv/bin/python -m unittest tests/test_smp_normalizer_resume.py`

Expected: runner round-trip and legacy behavior pass.

### Task 3: Register NewDev Exp13/Exp14 task family and fully mirrored Exp13 prior

**Files:**
- Modify: `src/smp/rl/tasks/body_velocity/__init__.py`
- Create: `src/smp/rl/tasks/body_velocity/body_velocity_exp14_new_dev_env_cfg.py`
- Create: `src/smp/rl/tasks/body_velocity/body_velocity_new_dev_exp13_env_cfg.py`
- Create: `scripts/run_new_dev_exp13_prepare_prior.sh`
- Create: `scripts/run_new_dev_exp13_groups4_6.sh`
- Create: `scripts/run_new_dev_exp14_groups1_9.sh`
- Create: `scripts/play_new_dev_exp14_groups1_9_wandb.sh`
- Modify: `tests/test_body_velocity_exp15.py`

- [x] **Step 1: Write failing builder and shell-contract tests**

Assert each new Exp13 G4--G6 task uses `SmpOnPolicyRunner`; assert Exp14 groups map separately to parent G4/G5/G6 and control/moving/static reward modes; assert scripts require a source run for Exp14 and use strict resume.

- [x] **Step 2: Run red tests**

Run: `./.venv/bin/python -m unittest tests/test_body_velocity_exp15.py`

Expected: new task ids/configs/scripts are absent.

- [x] **Step 3: Implement registrations and scripts**

Create an Exp13 prior builder that stages the Exp10 `loco + stop_static` sources,
creates every missing mirror including `stop_static_mirror.csv`, and validates every
staged CSV has a partner. Point exact Exp13 G4--G6 reruns at that resulting prior.
Create Exp14 groups 1--9: groups 1--3 control, 4--6 moving support-foot tilt, 7--9
static double-foot tilt; each maps to the same-numbered G4/G5/G6 parent. Use
low-intensity PPO CLI overrides and strict normalizer resume. Keep moving/static
additions separate.

- [x] **Step 4: Run green tests and shell syntax**

Run: `./.venv/bin/python -m unittest tests/test_body_velocity_exp15.py && bash -n scripts/run_new_dev_exp13_groups4_6.sh scripts/run_new_dev_exp14_groups1_9.sh scripts/play_new_dev_exp14_groups1_9_wandb.sh`

Expected: all pass.

### Task 4: Add strict four-file Exp16 prior preparation and policy configurations

**Files:**
- Create: `scripts/run_exp16_prepare_motebu_stop_static_prior.sh`
- Create: `src/smp/rl/tasks/body_velocity/body_velocity_exp16_env_cfg.py`
- Create: `scripts/run_exp16_groups1_3.sh`
- Create: `scripts/play_exp16_groups1_3_wandb.sh`
- Modify: `src/smp/rl/tasks/body_velocity/__init__.py`
- Modify: `scripts/experiment_launcher.py`
- Modify: `tests/test_body_velocity_exp15.py`

- [x] **Step 1: Write failing tests**

Add builder assertions for the new prior, x/y `[-1.5,1.5]`, yaw `[-2,2]`, zero probability `0.3`, and G4/G5/G6 reward mixtures. Add source-level shell checks that only `motebu.csv` and `stop_static.csv` are copied before mirroring, and that four CSV files are asserted.

- [x] **Step 2: Run red tests**

Run: `./.venv/bin/python -m unittest tests/test_body_velocity_exp15.py`

Expected: Exp16 tasks and scripts are absent.

- [x] **Step 3: Implement Exp16**

Make the prepare script stage the two named CSVs only, generate missing mirrors through `mirror_motion_data.py`, assert exactly the four expected names, convert/pretrain with Exp10-equivalent parameters, and copy the named prior. Build three from-scratch body-velocity configs from Exp13 G4/G5/G6, replace the prior and command ranges only. Register tasks and launcher mapping.

- [x] **Step 4: Run green verification**

Run: `./.venv/bin/python -m unittest tests/test_body_velocity_exp15.py && bash -n scripts/run_exp16_prepare_motebu_stop_static_prior.sh scripts/run_exp16_groups1_3.sh scripts/play_exp16_groups1_3_wandb.sh`

Expected: all pass.

### Task 5: Validate, document commands, and commit

**Files:**
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `task_plan.md`

- [x] **Step 1: Run full local verification**

Run: `MPLCONFIGDIR=/tmp/smp-mpl ./.venv/bin/python -m unittest tests/test_smp_normalizer_resume.py tests/test_body_velocity_exp15.py && ./.venv/bin/ruff check src/smp/rl scripts/run_new_dev_exp13_groups4_6.sh scripts/run_new_dev_exp14_groups1_9.sh scripts/run_exp16_prepare_motebu_stop_static_prior.sh && git diff --check`

Expected: all commands pass; no GPU training starts.

- [x] **Step 2: Record handoff**

Add exact commands in `findings.md`, state that Exp13 reruns create the first strict-resume parents, and state Exp16 prior preparation always includes four CSVs.

- [x] **Step 3: Commit new_dev implementation**

Run: `git add src/smp/rl tests scripts docs findings.md progress.md task_plan.md && git commit -m "feat: add resumable smp checkpoints and exp16 pipeline"`

Expected: one commit on `new_dev`; `reconstruct` remains unchanged.
