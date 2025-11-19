#!/bin/bash

# Configuration 
# ============================================
ENV="HalfCheetah-v2"
SEED=1
MAX_STEPS=100000
START_STEPS=5000
EVAL_FREQ=10000

# Sparsity settings
ACTOR_SPARSITY=0.25
CRITIC_SPARSITY=0.25
INITIAL_SPARSITY=0.75
DELTA=20000
ZETA=0.5
AWAKEN=0.25
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run Experiments
# ============================================

echo "Running experiments on $ENV with seed $SEED"
echo "============================================"

echo "Running NE-TD3"
# NE-TD3
python train.py \
    --exp_id ne_td3_${TIMESTAMP} \
    --env $ENV \
    --seed $SEED \
    --max_timesteps $MAX_STEPS \
    --start_timesteps $START_STEPS \
    --eval_freq $EVAL_FREQ \
    --actor_sparsity $ACTOR_SPARSITY \
    --critic_sparsity $CRITIC_SPARSITY \
    --initial_stl_sparsity $INITIAL_SPARSITY \
    --delta $DELTA \
    --zeta $ZETA \
    --awaken $AWAKEN \
    --stl_actor \
    --stl_critic &

echo "Running NE-SAC"
# NE-SAC
python train_sac.py \
    --exp_id ne_sac_${TIMESTAMP} \
    --env $ENV \
    --seed $SEED \
    --max_timesteps $MAX_STEPS \
    --start_timesteps $START_STEPS \
    --eval_freq $EVAL_FREQ \
    --actor_sparsity $ACTOR_SPARSITY \
    --critic_sparsity $CRITIC_SPARSITY \
    --initial_stl_sparsity $INITIAL_SPARSITY \
    --delta $DELTA \
    --zeta $ZETA \
    --awaken $AWAKEN \
    --stl_actor \
    --stl_critic &


wait
echo "Done!"