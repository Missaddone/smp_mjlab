# Experiment Launcher

Use `scripts/experiment_launcher.py` for Exp11, Exp12, and Exp13 policy jobs. The shared root-level [`wandb_run_registry.csv`](wandb_run_registry.csv) is the canonical mapping across servers; it contains only `exp`, `group`, and the bare W&B `run_id`.

Every planned group has a row. Starting a training job writes its generated run id into that row. On playback, the launcher reads the row; if it is blank, it searches the W&B `smp` project once using the experiment tag and group name, then writes the discovered id back to the CSV.

List supported experiments and group counts:

```bash
uv run scripts/experiment_launcher.py list
```

Start one selected group on a selected free GPU. This returns immediately after creating a tmux window:

```bash
uv run scripts/experiment_launcher.py train exp11 9 --gpu 3
uv run scripts/experiment_launcher.py train exp13 4 --gpu 5
```

The command prints and records the new W&B path. For Exp11, the script still uses its built-in Exp10 group14/group15 source checkpoint unless explicitly overridden. For Exp13 groups 7-18, the launcher automatically resolves the registered parent group (G4, G5, or G6) from `wandb_run_registry.csv` and W&B:

```bash
uv run scripts/experiment_launcher.py train exp11 25 --gpu 2 \
  --source-wandb-path robinbird-harbin-institute-of-technology/smp/<source_id> \
  --checkpoint-name model_9999.pt
uv run scripts/experiment_launcher.py train exp13 7 --gpu 2
```

Monitor all latest jobs and attach to the one tmux session:

```bash
uv run scripts/experiment_launcher.py status
uv run scripts/experiment_launcher.py attach
uv run scripts/experiment_launcher.py logs exp11 9 --follow
```

Replay a group without entering a W&B path:

```bash
uv run scripts/experiment_launcher.py play exp11 9 --gpu 0 --viewer viser
uv run scripts/experiment_launcher.py play exp13 4 --gpu 0 --viewer viser
```

For a one-time backfill of all existing runs in one experiment:

```bash
uv run scripts/experiment_launcher.py sync exp12
```

Local records and terminal logs remain in `.experiment_runs/`, which is intentionally ignored by Git. They are only for tmux/status/log inspection; playback always resolves through `wandb_run_registry.csv`.
