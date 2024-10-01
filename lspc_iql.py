import copy
from fsrl.utils import DummyLogger, WandbLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm.auto import trange

TensorBatch = List[torch.Tensor]



def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)



class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._costs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        self._cost_returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._costs[:n_transitions] = self._to_tensor(data["costs"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None]) + self._to_tensor(data["timeouts"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")



    def sample(self, batch_size: int) -> TensorBatch:
        indices = torch.randint(0, self._size, (batch_size,))
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        costs = self._costs[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, costs, next_states, dones]


def discounted_cumsum(x, gamma: float):
    """
    Calculate the discounted cumulative sum of x (can be rewards or costs).
    """
    cumsum = torch.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class VAE(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, latent_dim, act_lim, device="cpu"):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(obs_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, act_dim)

        self.act_lim = act_lim
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, obs, act):
        z = F.relu(self.e1(torch.cat([obs, act], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)
        return u, mean, std

    def decode(self, obs, z=None):
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).clamp(-0.5,
                                                                   0.5).to(self.device)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return self.act_lim * torch.tanh(self.d3(a))

    def decode_multiple(self, obs, z=None, num_decode=10):
        if z is None:
            z = torch.randn(
                (obs.shape[0], num_decode, self.latent_dim)).clamp(-0.5,
                                                                   0.5).to(self.device)

        a = F.relu(
            self.d1(
                torch.cat(
                    [obs.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)



class VAE_policy(VAE):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        beta: float = 0.5,
        clamper: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim, hidden_dim, latent_dim, max_action, device)
        self.beta = beta
        self.clamper = clamper

    def vae_kl_loss(self, state, action):
        recon, mean, std = self(state, action)
        KL_loss = -self.beta * 0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
        return KL_loss, recon
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        z = torch.randn((1, self.latent_dim), device=device).clamp(-self.clamper, self.clamper)
        return self.decode(state, z).cpu().data.numpy().reshape(1, -1)
    
    @torch.no_grad()
    def sample_n_actions(self, state: torch.Tensor, n_samples: int, clamp=True) -> torch.Tensor:
        if clamp:
            z = torch.randn((n_samples, self.latent_dim), device=self.device)
            z = torch.tanh(z) * self.clamper
        else:
            z = torch.randn((n_samples, self.latent_dim), device=self.device)
        state_expanded = state.expand(n_samples, -1)
        return self.decode(state_expanded, z), z

class SafetyEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, latent_dim: int = 32,
                 n_hidden: int = 2, max_action=0.2, log_std_min: float = -10.0,
                 log_std_max: float = 2.0):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden)]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> Normal:
        x = self.encoder(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())
    
    def sample_grad(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self(state)
        z = dist.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob
    

class SafeActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2, max_action: float = 1.0,
                 max_latent_action: float = 0.25, latent_dim: int = 32, vae_beta: float = 0.5, vae_clamper: float = 0.5,
                 log_std_min: float = -10.0, log_std_max: float = 2.0, device: str = "cuda"):
        super().__init__()
        self.actor = VAE_policy(state_dim, action_dim, max_action, hidden_dim, latent_dim, vae_beta, vae_clamper, device=device)
        self.safety_encoder = SafetyEncoder(state_dim, hidden_dim, latent_dim, n_hidden, max_latent_action, log_std_min, log_std_max)
        self.device = device

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent_action, _ = self.safety_encoder.sample_grad(state)
        decoded_action = self.actor.decode(state, latent_action)
        return decoded_action
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().reshape(1, -1)

    @torch.no_grad()
    def sample_n_actions(self, state: torch.Tensor, n_samples: int) -> torch.Tensor:
        latent_actions = self.safety_encoder(state).rsample((n_samples,)).squeeze(1)
        latent_actions = torch.tanh(latent_actions) * self.safety_encoder.max_action
        expanded_state = state.expand(n_samples, -1)
        decoded_actions = self.actor.decode(expanded_state, latent_actions)
        return decoded_actions, latent_actions
    


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
        n_hidden: int = 2, type: str = "reward"
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.type = type
        if type not in ["reward", "cost"]:
            raise ValueError("Invalid q function type")

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.type == "reward":
            return torch.min(*self.both(state, action))
        elif self.type == "cost":
            return torch.max(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


def LSPC_IQL_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    max_action = config["max_action"]
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    device = config["device"]
    actor_lr = config["actor_lr"]
    qf_lr = config["qf_lr"]
    vf_lr = config["vf_lr"]
    log_std_min = config["log_std_min"]
    log_std_max = config["log_std_max"]
    latent_dim = config["latent_dim"]
    vae_beta = config["vae_beta"]
    vae_clamper = config["vae_clamper"]
    max_latent_action = config["max_latent_action"]
    # networks
    actor = SafeActor(state_dim, action_dim, max_action=max_action, log_std_min=log_std_min, log_std_max=log_std_max,
                      latent_dim=latent_dim, vae_beta=vae_beta, vae_clamper=vae_clamper, max_latent_action=max_latent_action,
                      device=device).to(device)
    q_network = TwinQ(state_dim, action_dim).to(device)
    v_network = ValueFunction(state_dim).to(device)
    cost_q_network = TwinQ(state_dim, action_dim, type="cost").to(device)
    cost_v_network = ValueFunction(state_dim).to(device)
    # optimizers
    vae_actor_optimizer = torch.optim.Adam(actor.actor.parameters(), lr=actor_lr)
    safety_constraint_optimizer = torch.optim.Adam(actor.safety_encoder.parameters(), lr=actor_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=qf_lr)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=vf_lr)
    cost_q_optimizer = torch.optim.Adam(cost_q_network.parameters(), lr=qf_lr)
    cost_v_optimizer = torch.optim.Adam(cost_v_network.parameters(), lr=vf_lr)
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "vae_actor_optimizer": vae_actor_optimizer,
        "safety_constraint_optimizer": safety_constraint_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "cost_q_network": cost_q_network,
        "cost_q_optimizer": cost_q_optimizer,
        "cost_v_network": cost_v_network,
        "cost_v_optimizer": cost_v_optimizer,
        "device": device,
    }
    return kwargs


class LSPC_IQL_Trainer:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        vae_actor_optimizer: torch.optim.Optimizer,
        safety_constraint_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        cost_q_network: nn.Module,
        cost_q_optimizer: torch.optim.Optimizer,
        cost_v_network: nn.Module,
        cost_v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta_cost: float = 2.0,
        beta_reward: float = 3.0,
        discount: float = 0.99,
        tau: float = 0.005,
        exp_adv_max_cost: float = 1000.0,
        exp_adv_max_reward: float = 200.0,
        device: str = "cpu",
        logger: WandbLogger = DummyLogger(),
        episode_len: int = 300,
        safe_qc_vc_threshold: float = 0.02,
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.vae_actor_optimizer = vae_actor_optimizer
        self.safety_constraint_optimizer = safety_constraint_optimizer
        self.iql_tau = iql_tau
        self.cost_beta = beta_cost
        self.rew_beta = beta_reward
        self.discount = discount
        self.tau = tau
        self.exp_adv_max_cost = exp_adv_max_cost
        self.exp_adv_max_reward = exp_adv_max_reward
        self.logger = logger
        self.episode_len = episode_len
        self.safe_qc_vc_threshold = safe_qc_vc_threshold

        # cost critics
        self.cost_qf = cost_q_network
        self.cost_q_target = copy.deepcopy(self.cost_qf).requires_grad_(False).to(device)
        self.cost_q_optimizer = cost_q_optimizer
        self.cost_vf = cost_v_network
        self.cost_v_optimizer = cost_v_optimizer
        #

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

    def _update_cost_v(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        log_dict: Dict,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_cost_q = self.cost_q_target(states, actions)
            target_cost_q = torch.clamp(target_cost_q, min=0.0)
        cost_v = self.cost_vf(states)
        cost_v = torch.clamp(cost_v, min=0.0)
        cost_adv = -(target_cost_q - cost_v)

        # safety mask
        safety_threshold = self.safe_qc_vc_threshold
        safety_mask = (cost_v < safety_threshold).float() * (target_cost_q < safety_threshold).float()
        log_dict["safety_mask_mean"] = safety_mask.mean().item()
        log_dict["cost_v_mean"] = cost_v.mean().item()
        log_dict["cost_v_max"] = torch.max(cost_v).item()
        log_dict["cost_q_mean"] = target_cost_q.mean().item()
        log_dict["cost_q_max"] = torch.max(target_cost_q).item()
        log_dict["costs_mean"] = costs.mean().item()

        
        cost_v_loss = asymmetric_l2_loss(cost_adv, self.iql_tau)
        log_dict["cost_v_loss"] = cost_v_loss.item()
        self.cost_v_optimizer.zero_grad()
        cost_v_loss.backward()
        self.cost_v_optimizer.step()

        return cost_adv, safety_mask.detach()

    def _update_cost_q(
        self,
        next_cost_v: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        log_dict: Dict,
    ):
        targets = costs + (1.0 - dones.float()) * self.discount * next_cost_v.detach()
        cost_qs = self.cost_qf.both(states, actions)
        cost_q_loss = sum(F.mse_loss(q, targets) for q in cost_qs) / len(cost_qs)
        log_dict["cost_q_loss"] = cost_q_loss.item()
        self.cost_q_optimizer.zero_grad()
        cost_q_loss.backward()
        self.cost_q_optimizer.step()
        soft_update(self.cost_q_target, self.cost_qf, self.tau)
        

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
        safety_mask: torch.Tensor,
    ):
        exp_adv = torch.exp(self.cost_beta * adv.detach()).clamp(max=self.exp_adv_max_cost) * safety_mask
        kl_loss, action_out = self.actor.actor.vae_kl_loss(observations, actions)
        recon_losses = torch.sum((action_out - actions) ** 2, dim=1)
        policy_loss = torch.mean(exp_adv *(recon_losses + kl_loss))
        #
        max_exp_adv = torch.max(exp_adv)
        log_dict["weights/max_vae_exp_adv"] = max_exp_adv.item()
        min_exp_adv = torch.min(exp_adv)
        log_dict["weights/min_vae_exp_adv"] = min_exp_adv.item()
        mean_exp_adv = torch.mean(exp_adv)
        log_dict["weights/mean_vae_exp_adv"] = mean_exp_adv.item()
        log_dict["vae/kl_loss"] = torch.mean(kl_loss).item()
        log_dict["vae/recon_loss"] = torch.mean(recon_losses).item()
        log_dict["vae/awr_loss"] = policy_loss.item()
        log_dict["actor_loss"] = policy_loss.item()
        self.vae_actor_optimizer.zero_grad()
        policy_loss.backward()
        self.vae_actor_optimizer.step()

    def _update_safety_encoder(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.rew_beta * adv.detach()).clamp(max=self.exp_adv_max_reward)
        actions_out = self.actor(observations)
        bc_losses = torch.sum((actions_out - actions) ** 2, dim=1)
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["safety_cons/awr_loss"] = policy_loss.item()
        #
        max_exp_adv = torch.max(exp_adv)
        log_dict["weights/max_enc_exp_adv"] = max_exp_adv.item()
        min_exp_adv = torch.min(exp_adv)
        log_dict["weights/min_enc_exp_adv"] = min_exp_adv.item()
        mean_exp_adv = torch.mean(exp_adv)
        log_dict["weights/mean_enc_exp_adv"] = mean_exp_adv.item()
        #
        self.safety_constraint_optimizer.zero_grad()
        policy_loss.backward()
        self.safety_constraint_optimizer.step()


    
    def train_one_step(
        self, observations: torch.Tensor, next_observations: torch.Tensor, 
        actions: torch.Tensor, rewards: torch.Tensor, costs: torch.Tensor,
        dones: torch.Tensor
    ):
        self.total_it += 1
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
            next_cost_v = self.cost_vf(next_observations)
            next_cost_v = torch.clamp(next_cost_v, min=0.0)
            #
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        #
        # Update Cost critics
        cost_adv, safety_mask = self._update_cost_v(observations, actions, costs, log_dict)
        costs = costs.squeeze(dim=-1)
        self._update_cost_q(next_cost_v, observations, actions, costs, dones, log_dict)
        # Update CVAE policy
        self._update_policy(cost_adv, observations, actions, log_dict, safety_mask)
        # Update latent safety encoder policy
        self._update_safety_encoder(adv, observations, actions, log_dict)
        self.logger.store(**log_dict)
    


    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "vae_actor_optimizer": self.vae_actor_optimizer.state_dict(),
            "safety_constraint_optimizer": self.safety_constraint_optimizer.state_dict(),
            "cost_qf": self.cost_qf.state_dict(),
            "cost_vf": self.cost_vf.state_dict(),
            "cost_q_optimizer": self.cost_q_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.vae_actor_optimizer.load_state_dict(state_dict["vae_actor_optimizer"])
        self.safety_constraint_optimizer.load_state_dict(state_dict["safety_constraint_optimizer"])

        self.cost_qf.load_state_dict(state_dict["cost_qf"])
        self.cost_q_optimizer.load_state_dict(state_dict["cost_q_optimizer"])
        self.cost_q_target = copy.deepcopy(self.cost_qf)

        self.total_it = state_dict["total_it"]

    @torch.no_grad()
    def evaluate(self, env, eval_episodes: int) -> Tuple:
        self.actor.eval()
        rets, costs, lengths = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating LSPC-O Policy...", leave=False):
            ret, cost, length = self.rollout(env)
            rets.append(ret)
            costs.append(cost)
            lengths.append(length)
        self.actor.train()
        return np.mean(rets), np.mean(costs), np.mean(lengths), np.std(rets), np.std(costs), np.std(lengths)
    
    @torch.no_grad()
    def rollout(self, env) -> Tuple[float, float, int]:
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for _ in range(self.episode_len):
            act = self.actor.act(obs, self.device)
            act = act.flatten()
            obs_next, reward, terminated, truncated, info = env.step(act)
            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]
            if terminated or truncated:
                break
            obs = obs_next
        return episode_ret, episode_cost, episode_len
    

    @torch.no_grad()
    def evaluate_safe(self, env, eval_episodes, n_action_samples=32) -> Tuple:
        self.actor.eval()
        self.qf.eval()
        rets, costs, lengths = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating LSPC-S Policy ...", leave=False):
            ret, cost, length = self.rollout_safe(env)
            rets.append(ret)
            costs.append(cost)
            lengths.append(length)
        self.actor.train()
        self.qf.train()
        return np.mean(rets), np.mean(costs), np.mean(lengths), np.std(rets), np.std(costs), np.std(lengths)
    
    @torch.no_grad()
    def rollout_safe(self, env) -> Tuple[float, float, int]:
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for _ in range(self.episode_len):
            act = self.actor.actor.act(obs, self.device)
            act = act.flatten()
            obs_next, reward, terminated, truncated, info = env.step(act)
            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]
            if terminated or truncated:
                break
            obs = obs_next
        return episode_ret, episode_cost, episode_len
