# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
sys.path.append(r'../')
import numpy as np
import torch
import gymnasium as gym
import argparse
import os

from NE.utils import ReplayBuffer, show_sparsity
from SAC import SAC
from torch.utils.tensorboard import SummaryWriter
import json
from collections import deque
import copy


def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	step_seed = seed + 100
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = eval_env.reset(seed=step_seed)
		done = False
		truncated = False
		while not (done or truncated):
			action = policy.select_action(np.array(state), deterministic=True)
			state, reward, done, truncated, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"SAC Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_id", default='sac_experiment')
	parser.add_argument("--env", default='HalfCheetah-v4')
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--start_timesteps", default=25e3, type=int)
	parser.add_argument("--eval_freq", default=5e3, type=int)
	parser.add_argument("--max_timesteps", default=6e6, type=int)
	parser.add_argument("--batch_size", default=256, type=int)
	parser.add_argument("--discount", default=0.99)
	parser.add_argument("--tau", default=0.005)
	parser.add_argument("--Tamp", default=0.9, type=float)
	parser.add_argument("--hidden_dim", default=256, type=int)
	parser.add_argument("--static_actor", action='store_true', default=False)
	parser.add_argument("--static_critic", action='store_true', default=False)
	parser.add_argument("--actor_sparsity", default=0.25, type=float)
	parser.add_argument("--critic_sparsity", default=0.25, type=float)
	parser.add_argument("--initial_stl_sparsity", default=0.8, type=float)
	parser.add_argument("--delta", default=100, type=int)
	parser.add_argument("--zeta", default=0.5, type=float)
	parser.add_argument("--awaken", default=0.4, type=float)
	parser.add_argument("--recall", action='store_true', default=False)
	parser.add_argument("--auto_batch", action='store_true', default=True)
	parser.add_argument("--elastic", action='store_true', default=False)
	parser.add_argument("--grad_accumulation_n", default=1, type=int)
	parser.add_argument("--use_simple_metric", action='store_true', default=False)
	parser.add_argument("--complex_prune", action='store_true', default=False)
	parser.add_argument("--init_temperature", default=0.2, type=float, help="Initial entropy temperature (alpha)")
	parser.add_argument("--learnable_alpha", action='store_true', default=True)
	parser.add_argument("--alpha_lr", default=3e-4, type=float)
	parser.add_argument("--nstep", default=1, type=int)
	parser.add_argument("--delay_nstep", default=0, type=int)
	parser.add_argument("--buffer_max_size", default=int(1e6), type=int)
	parser.add_argument("--buffer_min_size", default=int(1e5), type=int)
	parser.add_argument("--use_dynamic_buffer", action='store_true', default=False)
	parser.add_argument("--buffer_threshold", default=0.2, type=float)
	parser.add_argument("--buffer_adjustment_interval", default=int(1e4), type=int)

	parser.add_argument("--grad_pruning", action='store_true', default=False)
	parser.add_argument("--grad_prune_alpha", default=0.99, type=float)

	args = parser.parse_args()

	args.T_end = (args.max_timesteps - args.start_timesteps)
	the_dir = 'results'
	root_dir = './'+the_dir+'/'+args.exp_id+'_'+args.env
	argsDict = copy.deepcopy(args.__dict__)
	del argsDict['seed']
	config_json=json.dumps(argsDict, indent=4)
	if not os.path.exists(root_dir):
		os.makedirs(root_dir)
	with open(root_dir+'/config.json','w') as file_json:
		file_json.write(config_json)
	if not os.path.exists("./"+the_dir):
		os.makedirs("./"+the_dir)

	print("---------------------------------------")
	print(f"SAC Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	exp_dir = root_dir+'/'+str(args.seed)
	tensorboard_dir = exp_dir+'/tensorboard/'
	model_dir = exp_dir+'/model/'

	os.makedirs(tensorboard_dir, exist_ok=True)
	os.makedirs(model_dir, exist_ok=True)

	torch.set_num_threads(1)

	env = gym.make(args.env)
	state, _ = env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	args.state_dim = state_dim
	args.action_dim = action_dim
	args.max_action = max_action

	writer = SummaryWriter(tensorboard_dir)
	policy = SAC(args, writer)

	if os.path.exists(model_dir+'actor') and os.path.exists(model_dir+'critic'):
		print("Loading saved SAC models...")
		checkpoint = torch.load(model_dir+'actor')
		policy.actor.load_state_dict(checkpoint)
		policy.critic.load_state_dict(torch.load(model_dir+'critic'))
		print("Models loaded successfully")

	if args.actor_sparsity > 0:
		print("Training a sparse SAC actor network")
		show_sparsity(policy.actor.state_dict())
	if args.critic_sparsity > 0:
		print("Training a sparse SAC critic network")
		show_sparsity(policy.critic.state_dict())

	replay_buffer = ReplayBuffer(state_dim, action_dim, args.buffer_max_size)

	evaluations = [eval_policy(policy, args.env, args.seed)]
	recent_eval = deque(maxlen=20)
	best_eval = np.mean(evaluations)

	done = False
	truncated = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	last_ac_fau = 0.0
	last_cr_fau = 0.0

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		if t < args.start_timesteps:
			action = env.action_space.sample()
			action_mean = action
		else:
			action_mean = policy.select_action(np.array(state))
			action = action_mean

		next_state, reward, done, truncated, info = env.step(action)
		done_bool = float(done or truncated)

		replay_buffer.add(state, action, next_state, reward, done_bool, action_mean, truncated)

		if args.use_dynamic_buffer and (t+1) % args.buffer_adjustment_interval == 0:
			if replay_buffer.size == replay_buffer.max_size:
				ind = (replay_buffer.ptr + np.arange(8*args.batch_size)) % replay_buffer.max_size
				replay_buffer.left_ptr = (replay_buffer.left_ptr + len(ind)) % replay_buffer.max_size
			elif replay_buffer.size > args.buffer_min_size and recent_eval and np.mean(recent_eval) > best_eval + args.buffer_threshold:
				replay_buffer.shrink()

		state = next_state
		episode_reward += reward

		if t >= args.start_timesteps:
			last_ac_fau, last_cr_fau = policy.train(replay_buffer, args.batch_size)

		if done or truncated:
			print(f"SAC Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			state, _ = env.reset()
			done = False
			truncated = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		if (t + 1) % args.eval_freq == 0:
			eval_reward = eval_policy(policy, args.env, args.seed)
			evaluations.append(eval_reward)
			recent_eval.append(eval_reward)
			writer.add_scalar('actor_FAU', last_ac_fau, t+1)
			writer.add_scalar('critic_FAU', last_cr_fau, t+1)
			writer.add_scalar('reward', eval_reward, t+1)
			if args.actor_sparsity > 0:
				writer.add_scalar('actor_sparsity', show_sparsity(policy.actor.state_dict(), to_print=False), t+1)
			if args.critic_sparsity > 0:
				writer.add_scalar('critic_sparsity', show_sparsity(policy.critic.state_dict(), to_print=False), t+1)
			if eval_reward > best_eval:
				best_eval = eval_reward
				torch.save(policy.actor.state_dict(), model_dir+'actor')
				torch.save(policy.critic.state_dict(), model_dir+'critic')

	print("Saving final SAC policy...")
	policy.save(model_dir + "sac_final.pth")
	writer.close()


if __name__ == "__main__":
	main()
