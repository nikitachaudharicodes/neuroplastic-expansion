import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from NE.STL_Scheduler import STL_Scheduler
from NE.utils import ReplayBuffer, get_W, show_sparsity
from modules_TD3 import MLPCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _copy_linear_params(dst_layer: nn.Linear, src_layer: nn.Linear):
	out_features = min(dst_layer.out_features, src_layer.out_features)
	in_features = min(dst_layer.in_features, src_layer.in_features)
	with torch.no_grad():
		dst_layer.weight[:out_features, :in_features].copy_(src_layer.weight[:out_features, :in_features])
		if dst_layer.bias is not None and src_layer.bias is not None:
			dst_layer.bias[:out_features].copy_(src_layer.bias[:out_features])


def _copy_layernorm_params(dst_ln: nn.LayerNorm, src_ln: nn.LayerNorm):
	dim = min(dst_ln.normalized_shape[0], src_ln.normalized_shape[0])
	with torch.no_grad():
		dst_ln.weight[:dim].copy_(src_ln.weight[:dim])
		dst_ln.bias[:dim].copy_(src_ln.bias[:dim])


class GaussianPolicy(nn.Module):
	LOG_SIG_MAX = 2
	LOG_SIG_MIN = -20

	def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
		super().__init__()
		self.max_action = max_action
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.mean_linear = nn.Linear(hidden_dim, action_dim)
		self.log_std_linear = nn.Linear(hidden_dim, action_dim)

	def forward(self, state):
		x = F.relu(self.l1(state))
		x = F.relu(self.l2(x))
		mean = self.mean_linear(x)
		log_std = torch.clamp(self.log_std_linear(x), self.LOG_SIG_MIN, self.LOG_SIG_MAX)
		return mean, log_std

	def sample(self, state):
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = Normal(mean, std)
		x_t = normal.rsample()
		y_t = torch.tanh(x_t)
		action = y_t * self.max_action

		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
		log_prob = log_prob.sum(dim=1, keepdim=True)
		mean_action = torch.tanh(mean) * self.max_action
		return action, log_prob, mean_action

	def forward_with_FAU(self, state):
		with torch.no_grad():
			h1 = F.relu(self.l1(state))
			total_1 = h1.numel()
			h2 = F.relu(self.l2(h1))
			total_2 = h2.numel()
			act_1 = (h1 > 0).sum().item()
			act_2 = (h2 > 0).sum().item()
		return (act_1 + act_2) / (total_1 + total_2 + 1e-8)

	def score_drop(self, state):
		score_drops = []
		with torch.no_grad():
			h1 = F.relu(self.l1(state))
			score_drops.append(h1)
			h2 = F.relu(self.l2(h1))
			score_drops.append(h2)
			mean = self.mean_linear(h2)
			score_drops.append(mean)
		return score_drops, [state.shape[-1], self.l1.out_features, self.l2.out_features]


class SAC(object):
	def __init__(self, args, writer):
		self.max_action = args.max_action
		self.discount = args.discount
		self.tau = args.tau
		self.Tamp = args.Tamp
		self.auto_batch = args.auto_batch
		self.awaken = args.awaken
		self.awaken_ = args.awaken * 0.9
		self.recall = args.recall
		self.elastic = args.elastic
		self.nstep = args.nstep
		self.delay_nstep = args.delay_nstep
		self.writer = writer
		self.tb_interval = int(args.T_end/1000)
		self.state_dim = args.state_dim
		self.action_dim = args.action_dim
		self.actor_hidden_dim = args.hidden_dim
		self.critic_hidden_dim = args.hidden_dim
		self.actor_lr = 3e-4
		self.critic_lr = 3e-4

		self.total_it = 0

		self.actor = GaussianPolicy(args.state_dim, args.action_dim, args.max_action, args.hidden_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		if self.elastic:
			self.actor_EMA = copy.deepcopy(self.actor)

		self.critic = MLPCritic(args.state_dim, args.action_dim, args.hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
		if self.elastic:
			self.critic_EMA = copy.deepcopy(self.critic)

		self.sparse_actor = (args.actor_sparsity > 0)
		self.sparse_critic = (args.critic_sparsity > 0)
		self.actor_sparsity = args.actor_sparsity
		self.critic_sparsity = args.critic_sparsity

		if self.sparse_actor:
			self.actor_pruner = STL_Scheduler(self.actor, self.actor_optimizer,
				static_topo=args.static_actor, sparsity=args.actor_sparsity,
				T_end=args.T_end, delta=args.delta, zeta=args.zeta,
				grad_accumulation_n=args.grad_accumulation_n,
				use_simple_metric=args.use_simple_metric,
				initial_stl_sparsity=args.initial_stl_sparsity,
				complex_prune=args.complex_prune)
			self.target_actor_W, _ = get_W(self.actor)
		else:
			self.actor_pruner = lambda *args, **kwargs: True

		if self.sparse_critic:
			self.critic_pruner = STL_Scheduler(self.critic, self.critic_optimizer,
				static_topo=args.static_critic, sparsity=args.critic_sparsity,
				T_end=args.T_end, delta=args.delta, zeta=args.zeta,
				grad_accumulation_n=args.grad_accumulation_n,
				use_simple_metric=args.use_simple_metric,
				initial_stl_sparsity=args.initial_stl_sparsity,
				complex_prune=args.complex_prune)
			self.target_critic_W, _ = get_W(self.critic_target)
		else:
			self.critic_pruner = lambda *args, **kwargs: True

		self.learnable_alpha = args.learnable_alpha
		if self.learnable_alpha:
			self.target_entropy = -args.action_dim
			self.log_alpha = torch.tensor(math.log(args.init_temperature)).to(device)
			self.log_alpha.requires_grad = True
			self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
		else:
			self.log_alpha = torch.tensor(math.log(args.init_temperature)).to(device)
			self.log_alpha.requires_grad = False

		self.adaptive_expansion = getattr(args, "adaptive_expansion", False)
		if self.adaptive_expansion:
			self._init_adaptive(args)

	def select_action(self, state, deterministic=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		if deterministic:
			_, _, action = self.actor.sample(state)
		else:
			action, _, _ = self.actor.sample(state)
		return action.cpu().data.numpy().flatten()

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def train(self, replay_buffer: ReplayBuffer, batch_size=256):
		current_batch = batch_size
		self.total_it += 1
		current_nstep = self.nstep if self.total_it >= self.delay_nstep else 1
		if self.total_it % self.tb_interval == 0:
			self.writer.add_scalar('current_nstep', current_nstep, self.total_it)

		state, action, next_state, reward, not_done, _, reset_flag = replay_buffer.sample(current_batch, current_nstep)
		with torch.no_grad():
			ac_fau = self.actor.forward_with_FAU(state[:, 0])
			cr_fau = self.critic.forward_with_FAU(state[:, 0], action[:, 0], None)
			gate = 1.0 / (1.0 + math.exp(self.awaken - cr_fau) * self.Tamp)
			if gate >= 0.60:
				gate = 0.4
		if self.auto_batch and cr_fau < self.awaken_:
			current_batch = max(32, int(current_batch / 2))
		if self.recall and cr_fau < self.awaken and random.uniform(0, 1) >= gate:
			state, action, next_state, reward, not_done, _, reset_flag = replay_buffer.recall_sample(int(current_batch), current_nstep)
		elif current_batch != batch_size:
			state, action, next_state, reward, not_done, _, reset_flag = replay_buffer.sample(int(current_batch), current_nstep)
		with torch.no_grad():
			ac_fau = self.actor.forward_with_FAU(state[:, 0])
			cr_fau = self.critic.forward_with_FAU(state[:, 0], action[:, 0], None)

		current_batch = state.shape[0]
		reward0 = reward[:, 0]
		state0 = state[:, 0]
		action0 = action[:, 0]

		with torch.no_grad():
			accum_reward = torch.zeros(reward[:, 0].shape).to(device)
			have_not_done = torch.ones(not_done[:, 0].shape).to(device)
			have_not_reset = torch.ones(not_done[:, 0].shape).to(device)
			modified_n = torch.zeros(not_done[:, 0].shape).to(device)
			for k in range(current_nstep):
				accum_reward += have_not_reset * have_not_done * (self.discount ** k) * reward[:, k]
				have_not_done *= torch.maximum(not_done[:, k], 1 - have_not_reset)
				if k == current_nstep - 1:
					break
				have_not_reset *= (1 - reset_flag[:, k])
				modified_n += have_not_reset
			modified_n = modified_n.type(torch.long)
			nstep_next_state = next_state[np.arange(state.shape[0]), modified_n[:, 0]]
			next_action, next_log_prob, _ = self.actor.sample(nstep_next_state)
			target_Q1, target_Q2 = self.critic_target(nstep_next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
			if current_nstep == 1:
				target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(target_Q.shape) * self.discount * target_Q
			else:
				target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(target_Q.shape) * self.discount ** (modified_n + 1) * target_Q

		current_Q1, current_Q2 = self.critic(state0, action0)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		end_grow = show_sparsity(self.critic.state_dict(), to_print=False) <= self.critic_sparsity
		if self.sparse_critic:
			if self.critic_pruner(end_grow, state0, action0, "critic"):
				self.critic_optimizer.step()
		else:
			self.critic_optimizer.step()

		pi, log_pi, _ = self.actor.sample(state0)
		q1_pi, q2_pi = self.critic(state0, pi)
		min_q_pi = torch.min(q1_pi, q2_pi)
		actor_loss = (self.alpha * log_pi - min_q_pi).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		end_grow = show_sparsity(self.actor.state_dict(), to_print=False) <= self.actor_sparsity
		if self.sparse_actor:
			if self.actor_pruner(end_grow, state0, None, "actor"):
				self.actor_optimizer.step()
		else:
			self.actor_optimizer.step()

		if self.learnable_alpha:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
		else:
			alpha_loss = torch.tensor(0.).to(device)

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if self.sparse_critic:
			for w, mask in zip(self.target_critic_W, self.critic_pruner.backward_masks):
				w.data *= mask

		if self.elastic:
			for param, EMA_param in zip(self.critic.parameters(), self.critic_EMA.parameters()):
				EMA_param.data.copy_(self.tau * param.data + (1 - self.tau) * EMA_param.data)
			for param, EMA_param in zip(self.actor.parameters(), self.actor_EMA.parameters()):
				EMA_param.data.copy_(self.tau * param.data + (1 - self.tau) * EMA_param.data)

		if self.sparse_actor:
			for w, mask in zip(self.target_actor_W, self.actor_pruner.backward_masks):
				w.data *= mask

		if self.tb_interval > 0 and self.writer is not None and self.total_it % self.tb_interval == 0:
			self.writer.add_scalar('sac/critic_loss', critic_loss.item(), self.total_it)
			self.writer.add_scalar('sac/actor_loss', actor_loss.item(), self.total_it)
			self.writer.add_scalar('sac/alpha', self.alpha.item(), self.total_it)

		if self.adaptive_expansion:
			self._maybe_expand(ac_fau, cr_fau)

		return ac_fau, cr_fau

	def _init_adaptive(self, args):
		self.adaptive_actor_threshold = args.adaptive_actor_threshold
		self.adaptive_critic_threshold = args.adaptive_critic_threshold
		self.adaptive_check_interval = args.adaptive_check_interval
		self.adaptive_expansion_size = args.adaptive_expansion_size
		self._last_adaptive_check = 0
		self.actor_expansions = 0
		self.critic_expansions = 0

	def _maybe_expand(self, ac_fau: float, cr_fau: float):
		if self.total_it < self.adaptive_check_interval:
			return
		if (self.total_it - self._last_adaptive_check) < self.adaptive_check_interval:
			return

		self._last_adaptive_check = self.total_it
		actor_expanded = False
		critic_expanded = False

		if ac_fau < self.adaptive_actor_threshold:
			self._expand_actor()
			actor_expanded = True
			self.actor_expansions += 1
		if cr_fau < self.adaptive_critic_threshold:
			self._expand_critic()
			critic_expanded = True
			self.critic_expansions += 1

		if self.writer is not None:
			self.writer.add_scalar('adaptive/actor_fau', ac_fau, self.total_it)
			self.writer.add_scalar('adaptive/critic_fau', cr_fau, self.total_it)
			if actor_expanded:
				self.writer.add_scalar('adaptive/actor_hidden_dim', self.actor_hidden_dim, self.total_it)
				self.writer.add_scalar('adaptive/actor_expansions', self.actor_expansions, self.total_it)
			if critic_expanded:
				self.writer.add_scalar('adaptive/critic_hidden_dim', self.critic_hidden_dim, self.total_it)
				self.writer.add_scalar('adaptive/critic_expansions', self.critic_expansions, self.total_it)

		if actor_expanded or critic_expanded:
			print(f"[Adaptive] Actor expanded={actor_expanded} Critic expanded={critic_expanded} at step {self.total_it}")

	def _expand_actor(self):
		new_hidden = self.actor_hidden_dim + self.adaptive_expansion_size
		new_actor = GaussianPolicy(self.state_dim, self.action_dim, self.max_action, new_hidden).to(device)
		self._copy_actor_weights(self.actor, new_actor)
		self.actor = new_actor
		self.actor_hidden_dim = new_hidden
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		if self.elastic:
			self.actor_EMA = copy.deepcopy(self.actor)

	def _expand_critic(self):
		new_hidden = self.critic_hidden_dim + self.adaptive_expansion_size
		new_critic = MLPCritic(self.state_dim, self.action_dim, new_hidden).to(device)
		self._copy_critic_weights(self.critic, new_critic)
		self.critic = new_critic
		self.critic_hidden_dim = new_hidden
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
		if self.elastic:
			self.critic_EMA = copy.deepcopy(self.critic)
		if self.sparse_critic:
			self.target_critic_W, _ = get_W(self.critic_target)

	def _copy_actor_weights(self, src_actor: GaussianPolicy, dst_actor: GaussianPolicy):
		_copy_linear_params(dst_actor.l1, src_actor.l1)
		_copy_linear_params(dst_actor.l2, src_actor.l2)
		_copy_linear_params(dst_actor.mean_linear, src_actor.mean_linear)
		_copy_linear_params(dst_actor.log_std_linear, src_actor.log_std_linear)

	def _copy_critic_weights(self, src_critic: MLPCritic, dst_critic: MLPCritic):
		_copy_linear_params(dst_critic.l, src_critic.l)
		_copy_linear_params(dst_critic.l1, src_critic.l1)
		_copy_linear_params(dst_critic.l2[0], src_critic.l2[0])
		_copy_layernorm_params(dst_critic.l2[1], src_critic.l2[1])
		_copy_linear_params(dst_critic.l3, src_critic.l3)

		_copy_linear_params(dst_critic.l4, src_critic.l4)
		_copy_linear_params(dst_critic.l5, src_critic.l5)
		_copy_linear_params(dst_critic.l6[0], src_critic.l6[0])
		_copy_layernorm_params(dst_critic.l6[1], src_critic.l6[1])
		_copy_linear_params(dst_critic.l7, src_critic.l7)

	def save(self, filename):
		torch.save({
			"actor": self.actor.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_target": self.critic_target.state_dict(),
			"log_alpha": self.log_alpha.detach().cpu()
		}, filename)

	def load(self, filename):
		checkpoint = torch.load(filename, map_location=device)
		self.actor.load_state_dict(checkpoint["actor"])
		self.critic.load_state_dict(checkpoint["critic"])
		if "critic_target" in checkpoint:
			self.critic_target.load_state_dict(checkpoint["critic_target"])
		if "log_alpha" in checkpoint:
			self.log_alpha.data.copy_(checkpoint["log_alpha"].to(device))
