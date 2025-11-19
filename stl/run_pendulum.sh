#!/bin/bash

echo "Quick Pendulum test"
echo "========================================"

echo "ne-td3 and ne-sac"
python train.py \
    --exp_id ne_td3 \
    --env Pendulum-v1 \
    --seed 1 \
    --max_timesteps 50000 \
    --start_timesteps 1000 \
    --eval_freq 5000 \
    --actor_sparsity 0.25 \
    --critic_sparsity 0.25 \
    --initial_stl_sparsity 0.75 \
    --delta 5000 \
    --zeta 0.5 \
    --awaken 0.25 \
    --stl_actor \
    --stl_critic &

python train_sac.py \
    --exp_id ne_sac \
    --env Pendulum-v1 \
    --seed 1 \
    --max_timesteps 50000 \
    --start_timesteps 1000 \
    --eval_freq 5000 \
    --actor_sparsity 0.25 \
    --critic_sparsity 0.25 \
    --initial_stl_sparsity 0.75 \
    --delta 5000 \
    --zeta 0.5 \
    --awaken 0.25 \
    --stl_actor \
    --stl_critic &

wait
echo "Done! Check results/"