#!/bin/bash
# SAC Consolidation-NE Experiments Runner

cd /home/ubuntu/neuroplastic-expansion
source venv/bin/activate
mkdir -p logs

echo "======================================"
echo "Starting SAC Consolidation-NE Experiments"
echo "======================================"
echo "Total: 4 experiments × ~1.5 hours each"
echo "Started at: $(date)"
echo "======================================"

# Test 1: Baseline
echo ""
echo "🚀 [1/4] Running: SAC Baseline Consolidation..."
python stl/train_sac.py \
  --exp_id sac_consolidation_baseline_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --learnable_alpha \
  --auto_batch \
  --recall \
  --max_timesteps 100000 \
  --consolidation_steps 1000 \
  --consolidation_lr_scale 1.0 \
  --consolidation_strategy uniform \
  > logs/sac_baseline_100k.log 2>&1
echo "✅ [1/4] SAC Baseline complete!"

# Test 2: Long + Low LR
echo ""
echo "🚀 [2/4] Running: SAC Long Consolidation + Low LR..."
python stl/train_sac.py \
  --exp_id sac_consolidation_long_lowlr_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --learnable_alpha \
  --auto_batch \
  --recall \
  --max_timesteps 100000 \
  --consolidation_steps 2000 \
  --consolidation_lr_scale 0.5 \
  --consolidation_strategy uniform \
  > logs/sac_long_lowlr_100k.log 2>&1
echo "✅ [2/4] SAC Long + Low LR complete!"

# Test 3: Recent Sampling
echo ""
echo "🚀 [3/4] Running: SAC Recent Experience Sampling..."
python stl/train_sac.py \
  --exp_id sac_consolidation_recent_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --learnable_alpha \
  --auto_batch \
  --recall \
  --max_timesteps 100000 \
  --consolidation_steps 1000 \
  --consolidation_lr_scale 1.0 \
  --consolidation_strategy recent \
  > logs/sac_recent_100k.log 2>&1
echo "✅ [3/4] SAC Recent Sampling complete!"

# Test 4: No Consolidation (Control)
echo ""
echo "🚀 [4/4] Running: SAC No Consolidation (Control)..."
python stl/train_sac.py \
  --exp_id sac_consolidation_none_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --learnable_alpha \
  --auto_batch \
  --recall \
  --max_timesteps 100000 \
  --consolidation_steps 0 \
  --consolidation_lr_scale 1.0 \
  --consolidation_strategy uniform \
  > logs/sac_none_100k.log 2>&1
echo "✅ [4/4] SAC No Consolidation complete!"

echo ""
echo "======================================"
echo "🎉 ALL SAC EXPERIMENTS COMPLETE!"
echo "Finished at: $(date)"
echo "======================================"
echo ""
echo "Results saved in: ./results/"
echo "Logs saved in: ./logs/"
echo ""
echo "To view results with TensorBoard:"
echo "  tensorboard --logdir results/"

