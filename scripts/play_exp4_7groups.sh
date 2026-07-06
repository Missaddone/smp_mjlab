# don't run this directly!!!
# copy and paste the commands into a terminal to run them

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 uv run scripts/play.py Smp-BodyVelocity-Exp4-Group1-G1 \
    --checkpoint-file logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-19_group1_mix_p07_l015_y015_kxy1_kyaw1/model_9999.pt \
    --video True \
    --video-length 1500

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 uv run scripts/play.py Smp-BodyVelocity-Exp4-Group2-G1 \
    --checkpoint-file logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-18_group2_mix_p07_l015_y015_kxy2_kyaw1/model_9999.pt \
    --video True \
    --video-length 1500

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 uv run scripts/play.py Smp-BodyVelocity-Exp4-Group3-G1 \
    --checkpoint-file logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-19_group3_mix_p07_l015_y015_kxy1_kyaw05/model_9999.pt \
    --video True \
    --video-length 1500

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 uv run scripts/play.py Smp-BodyVelocity-Exp4-Group4-G1 \
    --checkpoint-file logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-18_group4_sum_l05_y05_kxy1_kyaw05/model_9999.pt \
    --video True \
    --video-length 1500

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 uv run scripts/play.py Smp-BodyVelocity-Exp4-Group5-G1 \
    --checkpoint-file logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-18_group5_sum_l05_y05_kxy1_kyaw05_zero_neg_proj/model_9999.pt \
    --video True \
    --video-length 1500

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 uv run scripts/play.py Smp-BodyVelocity-Exp4-Group6-G1 \
    --checkpoint-file logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-18_group6_mix_p06_l02_y02_kxy1_kyaw1/model_9999.pt \
    --video True \
    --video-length 1500

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 uv run scripts/play.py Smp-BodyVelocity-Exp4-Group7-G1 \
    --checkpoint-file logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-18_group7_mix_p08_l01_y01_kxy1_kyaw1/model_9999.pt \
    --video True \
    --video-length 1500