#!/bin/bash
# Consolidation-NE Experiments Runner
# Run all 4 experiments in sequence

# Make sure we're in the right directory
cd /home/ubuntu/neuroplastic-expansion

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

echo "======================================"
echo "Starting Consolidation-NE Experiments"
echo "======================================"
echo "Total: 4 experiments × ~1.5 hours each"
echo "Started at: $(date)"
echo "======================================"

# Test 1: Baseline
echo ""
echo "🚀 [1/4] Running: Baseline Consolidation..."
python stl/train.py \
  --exp_id consolidation_baseline_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --max_timesteps 100000 \
  --consolidation_steps 1000 \
  --consolidation_lr_scale 1.0 \
  --consolidation_strategy uniform \
  > logs/baseline_100k.log 2>&1
echo "✅ [1/4] Baseline complete!"

# Test 2: Long + Low LR
echo ""
echo "🚀 [2/4] Running: Long Consolidation + Low LR..."
python stl/train.py \
  --exp_id consolidation_long_lowlr_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --max_timesteps 100000 \
  --consolidation_steps 2000 \
  --consolidation_lr_scale 0.5 \
  --consolidation_strategy uniform \
  > logs/long_lowlr_100k.log 2>&1
echo "✅ [2/4] Long + Low LR complete!"

# Test 3: Recent Sampling
echo ""
echo "🚀 [3/4] Running: Recent Experience Sampling..."
python stl/train.py \
  --exp_id consolidation_recent_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --max_timesteps 100000 \
  --consolidation_steps 1000 \
  --consolidation_lr_scale 1.0 \
  --consolidation_strategy recent \
  > logs/recent_100k.log 2>&1
echo "✅ [3/4] Recent Sampling complete!"

# Test 4: No Consolidation (Control)
echo ""
echo "🚀 [4/4] Running: No Consolidation (Control)..."
python stl/train.py \
  --exp_id consolidation_none_100k \
  --env HalfCheetah-v4 \
  --seed 0 \
  --stl_actor --stl_critic \
  --actor_sparsity 0.25 \
  --critic_sparsity 0.25 \
  --initial_stl_sparsity 0.8 \
  --uni \
  --max_timesteps 100000 \
  --consolidation_steps 0 \
  --consolidation_lr_scale 1.0 \
  --consolidation_strategy uniform \
  > logs/none_100k.log 2>&1
echo "✅ [4/4] No Consolidation complete!"

echo ""
echo "======================================"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "Finished at: $(date)"
echo "======================================"
echo ""
echo "Results saved in: ./results/"
echo "Logs saved in: ./logs/"
echo ""
echo "To view results with TensorBoard:"
echo "  tensorboard --logdir results/"