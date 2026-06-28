# 网络设定：
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export all_proxy=http://127.0.0.1:7897
export ALL_PROXY=http://127.0.0.1:7897



# pretrain model 训练
CUDA_VISIBLE_DEVICES=1 python scripts/smp_pretrain.py   --data-dir datasets/only_stand   --norm-stats-file datasets/g1_loco_norm_stats.npz   --log-dir logs/pretrain   --name g1_stand  --device cuda:0   --batch-size 1024   --num-epochs 4000   --save-interval 100   --log-interval 10

CUDA_VISIBLE_DEVICES=0 python scripts/smp_pretrain.py \
  --data-dir datasets/g1_walk_run \
  --norm-stats-file datasets/g1_walk_run_norm_stats.npz \
  --name g1_walk_run_local_norm \
  --device cuda:0 \
  --batch-size 1024 \
  --num-epochs 6000 \
  --num-timesteps 50 \
  --num-noise-samples 10 \
  --d-model 256 \
  --nhead 4 \
  --num-layers 2 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --train-split 0.9 \
  --save-interval 100 \
  --use-ema

# isaaclab任务仿真
CUDA_VISIBLE_DEVICES=0 \
python scripts/rsl_rl/train.py \
  --task Smp-G1-Steering-modified-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 10000

  # 导出ONNX
python scripts/rsl_rl/export_onnx.py \
  --checkpoint logs/rsl_rl/ \
  --output-dir /home/cyq/ONNX_OUTPUT  \
  --output-name smp_velocity_lafan_walk.onnx


# sim2sim手柄
  uv run mjpython sim2sim/run_velocity.py \
  --gamepad \
  --no-keyboard-controls \
  --gamepad-vx-scale 3.0 \
  --gamepad-vy-scale 3.0 \
  --gamepad-yaw-rate-scale 1.5


# csv2npz
  CUDA_VISIBLE_DEVICES=1 python scripts/smp_csv_to_npz.py \
  --input-dir datasets/lafan_walk_clips2_csv \
  --output-dir datasets/lafan_walk_clips2_npz \
  --window-size 10 \
  --stride 1 \
  --input-fps 120 \
  --output-fps 50 \
  --quat-order xyzw \
  --headless


  CUDA_VISIBLE_DEVICES=5 uv run python scripts/pretrain.py \
  --data-dir datasets/amp_lafan_walk_clips2_npz_mirrored \
  --norm-stats-file datasets/lafan_norm_stats.npz \
  --name amp_lafan_walk_clips2_mirrored_256 \
  --device cuda:0 \
  --batch-size 1024 \
  --num-epochs 50000 \
  --num-timesteps 50 \
  --num-noise-samples 10 \
  --d-model 128 \
  --nhead 4 \
  --num-layers 2 \
  --lr 0.0002 \
  --weight-decay 0.0001 \
  --train-split 0.9 \
  --save-interval 100 \
  --use-ema
  --no-use-wandb

CUDA_VISIBLE_DEVICES=6 python scripts/rsl_rl/train.py \
  --task Smp-G1-BodyVelocity-LafanRun-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 10000


CUDA_VISIBLE_DEVICES=6 uv run scripts/train.py Smp-Forward-G1 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 4096

CUDA_VISIBLE_DEVICES=3  uv run scripts/train.py Smp-BodyVelocity-G1 \
  --agent.max-iterations 10000 \
  --env.scene.num-envs 4096

  CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-BodyVelocity-FootRegularized-G1 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 4096


  CUDA_VISIBLE_DEVICES=4 uv run scripts/train.py Smp-ForwardBackward-G1 \
  --agent.max-iterations 10000 \
  --env.scene.num-envs 4096


  uv run scripts/generate_viz.py   --ckpt-path datasets/lafan_run_all_norm.pt   --device cuda:0   --fps 50


uv run scripts/play.py  Smp-BodyVelocity-LafanWalk-G1 --num-envs=16 --checkpoint-file logs/model_2500.pt  --video True --video-length 800   --video-width 1280   --video-height 720


uv run python scripts/mirror_motion_data.py \
  --input-dir datasets/amp_loco_clips_npz \
  --output-dir datasets/amp_loco_clips_npz_mirrored \
  --include-original \
  --no-csv \
  --npz


uv run python scripts/mirror_motion_data.py \
  --input-dir datasets/motion_data_csv \
  --output-dir datasets/motion_data_csv_mirrored \
  --include-original \
  --csv \
  --no-npz


  uv run scripts/csv_to_npz.py   --input-dir datasets/amp_loco_clips2_csv_mirrored   --output-dir datasets/amp_loco_clips2_npz_mirrored --input-fps 120 --output-fps 50


CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-BodyForwardBackward-G1 \
--agent.max-iterations 10000 \
--env.scene.num-envs=4096


CUDA_VISIBLE_DEVICES=3 uv run scripts/train.py Smp-BodyVelocity-Sum-G1 \
  --agent.max-iterations 10000 \
  --env.scene.num-envs 4096