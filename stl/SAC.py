import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from NE.STL_Scheduler import STL_Scheduler
from NE.utils import ReplayBuffer, get_W, show_sparsity
from modules_TD3 import MLPCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class SAC(object):
	def __init__(self, args, writer):
		self.max_action = args.max_action
		self.discount = args.discount
		self.tau = args.tau
		self.writer = writer
		self.tb_interval = int(args.T_end/1000)

		self.total_it = 0

		self.actor = GaussianPolicy(args.state_dim, args.action_dim, args.max_action, args.hidden_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = MLPCritic(args.state_dim, args.action_dim, args.hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

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
		self.total_it += 1
		state, action, next_state, reward, not_done, _, _ = replay_buffer.sample(batch_size)
		state = state[:,0]
		action = action[:,0]
		next_state = next_state[:,0]
		reward = reward[:,0]
		not_done = not_done[:,0]

		with torch.no_grad():
			next_action, next_log_prob, _ = self.actor.sample(next_state)
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
			target_Q = reward + not_done * self.discount * target_Q

		current_Q1, current_Q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		end_grow = show_sparsity(self.critic.state_dict(), to_print=False) <= self.critic_sparsity
		if self.sparse_critic:
			if self.critic_pruner(end_grow, state, action, "critic"):
				self.critic_optimizer.step()
		else:
			self.critic_optimizer.step()

		pi, log_pi, _ = self.actor.sample(state)
		q1_pi, q2_pi = self.critic(state, pi)
		min_q_pi = torch.min(q1_pi, q2_pi)
		actor_loss = (self.alpha * log_pi - min_q_pi).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		end_grow = show_sparsity(self.actor.state_dict(), to_print=False) <= self.actor_sparsity
		if self.sparse_actor:
			if self.actor_pruner(end_grow, state, None, "actor"):
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

		if self.tb_interval > 0 and self.writer is not None and self.total_it % self.tb_interval == 0:
			self.writer.add_scalar('sac/critic_loss', critic_loss.item(), self.total_it)
			self.writer.add_scalar('sac/actor_loss', actor_loss.item(), self.total_it)
			self.writer.add_scalar('sac/alpha', self.alpha.item(), self.total_it)

		return critic_loss.item(), actor_loss.item(), self.alpha.item()

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
