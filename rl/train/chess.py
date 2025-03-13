import collections
import datetime
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pettingzoo import AECEnv
from torch.distributions import Normal
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipMultiheadAttentionPoolingHead,
)

from rl.zoo.chess.chess import ChessEnv


# ==============================
# Custom RL Configuration
# ==============================
class RLConfig(PretrainedConfig):
    model_type: str = "fixed-width-attention"
    state_tokens: str = "\0 -/0123456789BKNPQRabcdefghknpqrw"
    max_state_length: int = 100
    max_mask_count: int = 256
    num_actions: Union[int, Tuple[int]] = (64, 64, 8)

    hidden_size: int = 256
    num_hidden_layers: int = 4
    _attn_implementation: str = "sdpa"
    layer_norm_eps: float = 1e-6
    num_attention_heads: int = 8
    intermediate_size: int = 512
    hidden_act: str = "gelu_pytorch_tanh"
    attention_dropout: float = 0.0

    policy_loss_weight: float = 1.0
    policy_valid_loss_weight: float = 0.1
    value_loss_weight: float = 0.0

    reward_default: float = 0.0
    reward_win: float = 1.0
    reward_draw: float = -0.01
    reward_loss: float = -1.0


class RLTokenizer(PreTrainedTokenizer):
    def __init__(self, config, **kwargs):
        self.state_tokens = config.state_tokens
        self.vocab = {token: idx for idx, token in enumerate(self.state_tokens)}
        self.ids_to_tokens = {idx: token for idx, token in enumerate(self.state_tokens)}
        self.unk_token = None
        self.pad_token = "\0"
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):
        return [ch for ch in text if ch in self.vocab]

    def _convert_token_to_id(self, token):
        if token in self.vocab:
            return self.vocab[token]
        elif self.unk_token is not None:
            return self.vocab.get(self.unk_token)
        else:
            raise ValueError(f"Token '{token}' not found in vocabulary.")

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        **kwargs,
    ):
        tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return self.convert_tokens_to_string(tokens)

    def _add_tokens(self, new_tokens, special_tokens=False):
        for i, t in enumerate(new_tokens, start=self.vocab_size):
            self.vocab[t] = i
            self.ids_to_tokens[i] = t


class RLPretrainedModel(PreTrainedModel):

    def _init_weights(self, module):
        @torch.no_grad()
        def normalize_dim(tensor: torch.Tensor, scale: float = 1.0, dim: int = -1):
            return tensor.normal_(0, scale).div_(tensor.size(dim) ** 0.5)

        if isinstance(module, PolicyNetwork):
            normalize_dim(module.positional_embedding)
            normalize_dim(module.projection, 0.1)
            # module.logit_scale.data.fill_(0.0)
        if isinstance(module, ValueNetwork):
            nn.init.constant_(module.fc.weight, 0.0)
            nn.init.constant_(module.fc.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            normalize_dim(module.weight)
        elif isinstance(module, SiglipAttention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, SiglipMLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, SiglipMultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)

    @classmethod
    def rms_norm(cls, x: torch.Tensor, dim=-1):
        return x / (x.norm(dim=dim, keepdim=True) + 1e-6)

    @classmethod
    def dummy_hidden(cls, batch_size):
        return torch.tensor(1.0).expand(batch_size, 1)


class PolicyNetwork(RLPretrainedModel):
    def __init__(self, config: RLConfig):
        super().__init__(config)

        self.actions = config.num_actions
        query_size = np.prod(self.actions[:1])
        proj_size = np.prod(self.actions[1:])

        self.ff_length = 1 << int(query_size + config.max_state_length - 1).bit_length()
        self.embedding = nn.Embedding(len(config.state_tokens), config.hidden_size)
        self.positional_embedding = nn.Parameter(
            torch.empty([self.ff_length, config.hidden_size])
        )
        self.projection = nn.Parameter(torch.empty([proj_size, config.hidden_size]))
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_layernorm = nn.RMSNorm(config.hidden_size)

        self.post_init()

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        """
        x: shape [B, T, D]
        hidden: optional shape [B, D]
        """
        B, T, _ = x.shape
        x = x.flatten(0, 1)  # [BT, D]
        x = self.embedding(x)

        pad_shape = list(x.shape)
        pad_shape[1] = self.ff_length - self.config.max_state_length

        x = torch.cat([x.new_zeros(pad_shape), x], 1)
        x = x + self.positional_embedding[None]
        for encoder_layer in self.layers:
            x = encoder_layer(x, attention_mask=None)[0]
        x = x[..., : self.actions[0], :]
        query = self.post_layernorm(x)  # [B x T, A1, d]
        key = self.projection  # [A2 x A3, d]

        logits = torch.einsum("bid,jd->bij", query, key)
        logits = F.log_softmax(logits.flatten(1), -1, dtype=torch.float32).reshape(
            B, T, *self.actions
        )
        return logits, self.dummy_hidden(B)

    def forward_single(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ):
        """ """
        x = x.unsqueeze(1)
        logits, hidden = self.forward(x, hidden)
        return logits.squeeze(1), hidden


class ValueNetwork(RLPretrainedModel):
    def __init__(self, config: RLConfig):
        super().__init__(config)

        self.actions = config.num_actions
        self.ff_length = 1 << (config.max_state_length - 1).bit_length()
        self.positional_embedding = nn.Parameter(
            torch.empty([self.ff_length, config.hidden_size])
        )
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_layernorm = nn.RMSNorm(config.hidden_size)
        self.head = SiglipMultiheadAttentionPoolingHead(config)
        self.fc = nn.Linear(config.hidden_size, 2)  # Output: [mean, log_std]

        self.post_init()

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, state_dim).
            prev_action: Tensor of shape (batch, seq_len, num_actions).
            hidden: Optional GRU hidden state.
        Returns:
            mean, std: Tensors of shape (batch, seq_len).
            hidden: GRU hidden state.
        """
        B, T, _ = x.shape
        x = x.flatten(0, 1)  # [BT, D]
        x = self.embedding(x) + self.positional_embedding[None, None]
        for encoder_layer in self.layers:
            x = encoder_layer(x)[0]
        x = self.post_layernorm(x)
        x = self.head(x)
        x = self.fc(x)
        mean, log_std = x.reshape(B, T, 2).unbind(-1)
        std = F.softplus(log_std)
        return mean, std, self.dummy_hidden(B)

    def forward_single(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ):
        x = x.unsqueeze(1)
        mean, std, hidden = self.forward(x, hidden)
        return mean.squeeze(1), std.squeeze(1), hidden


# ==============================
# RL Model (Policy and Value Networks)
# ==============================
class RLModel(RLPretrainedModel):
    config_class = RLConfig

    def __init__(self, config: RLConfig):
        super().__init__(config)
        self.config = config
        self.policy_net = PolicyNetwork(config)
        self.value_net = ValueNetwork(config)

        self.post_init()

    def forward(
        self,
        input_states: torch.Tensor,
        mode: str = "train",
        hidden_policy: Optional[torch.Tensor] = None,
        hidden_value: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mode: One of "train", "trajectory", "explore", or "evaluate".
            input_states: For "train", a tensor of shape (B, T, state_dim); for "explore" or "evaluate", (B, state_dim).
        Returns:
            A dictionary containing logits and/or value estimates.
        """
        if mode == "train":
            logits, _ = self.policy_net(input_states, hidden_policy)
            value_mean, value_std, _ = self.value_net(input_states, hidden_value)
            return {"logits": logits, "value_mean": value_mean, "value_std": value_std}
        elif mode == "trajectory":
            logits, _ = self.policy_net(input_states, hidden_policy)
            return {"logits": logits}
        elif mode == "explore":
            logits, new_hidden_policy = self.policy_net.forward_single(
                input_states, hidden_policy
            )
            return {"logits": logits, "hidden_policy": new_hidden_policy}
        elif mode == "evaluate":
            value_mean, value_std, new_hidden_value = self.value_net.forward_single(
                input_states, hidden_value
            )
            return {
                "value_mean": value_mean,
                "value_std": value_std,
                "hidden_value": new_hidden_value,
            }
        else:
            raise ValueError("Invalid mode")


# ==============================
# Trajectory Dataset
# ==============================
class RLTrajectoryDataset(torch.utils.data.Dataset):
    """
    Each item is a trajectory dict with the following keys:
      - state: Tensor of shape (T, state_dim)
      - action: Tensor of shape (T,) (integer type)
      - action_logit: Tensor of shape (T,)
      - reward: Tensor of shape (T,)
      - z_target: Tensor of shape (T,) (cumulative reward without discounting)
      - intrinsic_reward: Tensor of shape (T-1,)
    """

    def __init__(self, trajectories: List[Dict[str, Any]]):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


# ==============================
# Data Collator: Pad trajectories to same length
# ==============================
def trajectory_data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    states = [traj["state"] for traj in features]
    actions = [traj["action"] for traj in features]
    action_masks = [traj["action_mask"] for traj in features]
    action_logits = [traj["action_logit"] for traj in features]
    rewards = [traj["reward"] for traj in features]
    z_targets = [traj["z_target"] for traj in features]
    intrinsic_rewards = [traj["intrinsic_reward"] for traj in features]
    opt = dict(batch_first=True)

    states_padded = torch.nn.utils.rnn.pad_sequence(states, **opt)
    actions_padded = torch.nn.utils.rnn.pad_sequence(actions, **opt)
    action_logits_padded = torch.nn.utils.rnn.pad_sequence(action_logits, **opt)
    rewards_padded = torch.nn.utils.rnn.pad_sequence(rewards, **opt)
    z_targets_padded = torch.nn.utils.rnn.pad_sequence(z_targets, **opt)
    intrinsic_rewards_padded = torch.nn.utils.rnn.pad_sequence(intrinsic_rewards, **opt)
    action_mask_padded = torch.nn.utils.rnn.pad_sequence(action_masks, **opt)
    mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(len(traj["state"]), dtype=torch.bool) for traj in features], **opt
    )

    return {
        "state": states_padded,
        "action": actions_padded,
        "action_mask": action_mask_padded,
        "action_logit": action_logits_padded,
        "reward": rewards_padded,
        "z_target": z_targets_padded,
        "intrinsic_reward": intrinsic_rewards_padded,
        "mask": mask,
    }


class RewardPostProcessor:
    def __init__(self, config: RLConfig):
        self.default = config.reward_default
        self.loss = config.reward_loss
        self.draw = config.reward_draw
        self.win = config.reward_win
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        pass

    def process(self, reward: float, termination: bool, truncation: bool) -> float:
        self.num_steps += 1
        value = 0

        if not termination:
            value += self.default
        if reward == 1.0:
            value += self.win
        if reward == 0.0 and termination:
            value += self.draw
        if reward == -1.0:
            value += self.loss
        return value


class TrajectoryState(Enum):
    FIRST_OBSERVATION = 0
    WAITING_FOR_LAST = 1
    WAITING_FOR_STEP = 2
    FINISHED = 3
    COLLECTED = 4


class TrajectoryManager:
    def __init__(
        self,
        env: AECEnv,
        tokenizer: RLTokenizer,
        reward_processor: RewardPostProcessor,
        config: RLConfig,
    ):
        # TODO: Remove tokenizer and reward_processor dependencies from manager
        self.env = env
        self.tokenizer = tokenizer
        self.reward_processor = reward_processor
        self.config = config

        self.t_state = []
        self.t_reward = []
        self.t_action_mask = []
        self.t_action = []
        self.t_action_logit = []
        self.t_terminated = []
        self.t_truncated = []

        self._state = None
        self._mask = None
        self._creward = None
        self._prev_creward = 0

        self.step_trigger = False
        self.state = TrajectoryState.FIRST_OBSERVATION

    @torch.no_grad()
    def last(self):
        match self.state:
            case TrajectoryState.FIRST_OBSERVATION | TrajectoryState.WAITING_FOR_LAST:
                obs, self._creward, terminated, truncated, info = self.env.last()
                reward = self._creward - self._prev_creward
                r = self.reward_processor.process(reward, terminated, truncated)
                if self.state == TrajectoryState.WAITING_FOR_LAST:
                    self.t_reward.append(r)
                    self.t_terminated.append(terminated)
                    self.t_truncated.append(truncated)
                if terminated or truncated:
                    self.state = TrajectoryState.FINISHED
                    state = None
                    print(f"env : {info=}")
                else:
                    self.state = TrajectoryState.WAITING_FOR_STEP
                    self._state = self.tokenizer.encode(
                        obs["fen"],
                        return_tensors=None,
                        max_length=self.config.max_state_length,
                        padding_side="right",
                        padding="max_length",
                    )
                    state = torch.tensor(self._state)
                    self._mask = torch.sparse_coo_tensor(
                        indices=torch.tensor(
                            obs["allowed_actions"], dtype=torch.long
                        ).t(),
                        values=torch.ones(
                            len(obs["allowed_actions"]), dtype=torch.bool
                        ),
                        size=self.config.num_actions,
                        dtype=torch.bool,
                    )
                return state, (terminated or truncated)
            case _:
                raise RuntimeError(
                    "last() must be called only when state is FIRST_OBSERVATION or WAITING_FOR_LAST."
                )

    @torch.no_grad()
    def step(self, logits):
        match self.state:
            case TrajectoryState.WAITING_FOR_STEP:
                self.step_trigger = True
                self._prev_creward = self._creward

                allowed_indices = self._mask._indices()  # shape: [ndim, n_allowed]
                allowed_logits = logits[tuple(allowed_indices)]  # shape: [n_allowed]

                log_probs = torch.log_softmax(allowed_logits, dim=0)
                probs = torch.softmax(allowed_logits, dim=0)
                chosen_idx = torch.multinomial(probs, num_samples=1).item()
                action = allowed_indices[:, chosen_idx].tolist()
                self.env.step(action)

                self.t_state.append(self._state)
                self.t_action_mask.append(self._mask)
                self.t_action.append(action)
                self.t_action_logit.append(log_probs[chosen_idx].item())

                self.state = TrajectoryState.WAITING_FOR_LAST
            case _:
                raise RuntimeError(
                    "step() must be called only when state is WAITING_FOR_STEP."
                )

    @torch.no_grad()
    def collect(self):
        match self.state:
            case TrajectoryState.FINISHED:
                self.env.step(None)
                reward_tensor = torch.tensor(self.t_reward, dtype=torch.float)
                collected_data = dict(
                    state=torch.tensor(self.t_state, dtype=torch.long),
                    action_mask=torch.stack(self.t_action_mask),
                    action=torch.tensor(self.t_action, dtype=torch.long),
                    action_logit=torch.tensor(self.t_action_logit, dtype=torch.float),
                    reward=reward_tensor,
                    z_target=reward_tensor.flip(0).cumsum(0).flip(0),
                    intrinsic_reward=reward_tensor[:-1],
                    terminated=torch.tensor(self.t_terminated, dtype=torch.bool),
                    truncated=torch.tensor(self.t_truncated, dtype=torch.bool),
                )
                self.state = TrajectoryState.COLLECTED
                return collected_data
            case TrajectoryState.COLLECTED:
                raise RuntimeError(
                    "collect() has already been called; trajectory has been collected."
                )
            case _:
                raise RuntimeError(
                    "collect() can only be called when state is FINISHED."
                )


AgentType = Any


@dataclass(frozen=True)
class ModelVariable:
    model: RLPretrainedModel
    tokenizer: RLTokenizer
    reward_processor: RewardPostProcessor


@dataclass
class AgentConfig:
    model_variable: ModelVariable
    traj: TrajectoryManager
    hidden: torch.Tensor


@dataclass
class EnvironmentVariable:
    env: AECEnv
    agent_iter: Iterator
    agents: Dict[AgentType, AgentConfig]

    @classmethod
    def build(cls, ENV: Type[AECEnv], mapping: Dict[AgentType, List[ModelVariable]]):
        env = ENV()
        env.reset()
        agents_config: Dict[Any, AgentConfig] = {}
        for agent in env.possible_agents:
            model_var = random.choice(mapping[agent])
            agents_config[agent] = AgentConfig(
                model_variable=model_var,
                traj=TrajectoryManager(
                    env=env,
                    tokenizer=model_var.tokenizer,
                    reward_processor=model_var.reward_processor,
                    config=model_var.model.config,
                ),
                hidden=model_var.model.dummy_hidden(1)[0],
            )
        return cls(
            env=env,
            agent_iter=iter(env.agent_iter()),
            agents=agents_config,
        )


@torch.no_grad()
def collect_trajectory(env_var: List[EnvironmentVariable]) -> List[Dict[str, Any]]:
    final_trajectories = []
    while True:
        task_mapping: Dict[ModelVariable, List[Tuple[torch.Tensor, AgentConfig]]] = (
            collections.defaultdict(list)
        )
        for ev in env_var:
            try:
                agent = next(ev.agent_iter)
            except StopIteration:
                continue

            agent_config = ev.agents[agent]
            state, done = agent_config.traj.last()
            if done:
                final_trajectories.append(agent_config.traj.collect())
            else:
                task_mapping[agent_config.model_variable].append((state, agent_config))

        if not task_mapping:
            break

        for model_var, tasks in task_mapping.items():
            device = next(model_var.model.parameters()).device
            dtype = next(model_var.model.parameters()).dtype

            state_, ac_ = zip(*tasks)
            outputs = model_var.model(
                torch.stack(state_).to(device),
                mode="explore",
                hidden_policy=torch.stack([ac.hidden for ac in ac_]),
            )
            for ac, l, h in zip(ac_, outputs["logits"], outputs["hidden_policy"]):
                ac.traj.step(l)
                ac.hidden = h
    return final_trajectories


# ==============================
# Custom Trainer: Override compute_loss (process trajectories)
# ==============================
class RLTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.first_batch = True
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        """
        Args:
            inputs: A dict with keys: states, actions, action_logits, rewards, z_targets, intrinsic_rewards, mask.
        Returns:
            total_loss and optionally a dict with policy_loss and value_loss.
        """
        state = inputs["state"]
        action = inputs["action"]
        action_mask = inputs["action_mask"]
        action_logit = inputs["action_logit"]
        reward = inputs["reward"]
        z_target = inputs["z_target"]
        intrinsic_reward = inputs["intrinsic_reward"]
        mask = inputs["mask"]

        B, C, _ = action.shape
        i_idx = torch.arange(B).view(B, 1).expand(B, C)
        j_idx = torch.arange(C).view(1, C).expand(B, C)
        outputs = model(state, mode="trajectory")

        masked_logits = F.log_softmax(
            outputs["logits"].masked_fill(~action_mask, float("-inf")).flatten(2), -1
        ).reshape_as(action_mask)
        policy_logit = masked_logits[i_idx, j_idx, *action.unbind(-1)]
        log_rho = (policy_logit - action_logit.detach()).masked_fill(~mask, 0).cumsum(1)
        log_rho_offset = log_rho.detach()[mask].logsumexp(0)
        scaled_rho = (log_rho - log_rho_offset).exp()
        policy_loss = (scaled_rho * reward)[mask].sum().mul(-1)

        if self.config.policy_valid_loss_weight:
            policy_valid_loss = (
                outputs["logits"]
                .masked_fill(~action_mask, float("-inf"))
                .flatten(-len(self.config.num_actions))
                .logsumexp(-1)[mask]
                .mean()
                .mul(-1)
            )
        else:
            policy_valid_loss = 0

        if self.config.value_loss_weight:
            outputs = model(state, mode="evaluate")
            value_mean = outputs["value_mean"]
            value_std = outputs["value_std"]
            normal_dist = Normal(value_mean, value_std + 1e-6)
            log_prob_value = normal_dist.log_prob(z_target)
            value_loss = log_prob_value[mask].mean().mul(-1)
        else:
            value_loss = 0

        total_loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.policy_valid_loss_weight * policy_valid_loss
            + self.config.value_loss_weight * value_loss
        )
        print(f"{torch.is_tensor(policy_loss) and policy_loss.item()}")
        print(f"{torch.is_tensor(policy_valid_loss) and policy_valid_loss.item()}")
        print(f"{torch.is_tensor(value_loss) and value_loss.item()}")

        if self.first_batch:
            self.first_batch = False

        if return_outputs:
            return total_loss, {
                "policy_loss": policy_loss.item(),
                "policy_valid_loss": policy_valid_loss.item(),
                "value_loss": value_loss.item(),
            }
        else:
            return total_loss


# ==============================
# Logging Callback
# ==============================
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "policy_loss" in logs and "value_loss" in logs:
            print(
                f"Step {state.global_step} - Policy Loss: {logs['policy_loss']:.4f}, Value Loss: {logs['value_loss']:.4f}"
            )


# ==============================
# Experiment Run Loop
# ==============================
def run_experiment(
    ENV: Type[AECEnv],
    config: RLConfig,
    batch_size: int = 16,
    rollout: int = 128,
    training_epoch: int = 8,
    training_iteration: int = 100,
    trial: str = "",
    **kwargs,
):
    # single agent self play
    env_name = ENV.__name__
    env = ENV()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RLModel(config).to(device).bfloat16()
    model_var_ = [
        ModelVariable(model, RLTokenizer(config), RewardPostProcessor(config))
    ]
    model_mapping = {ag: model_var_ for ag in env.possible_agents}

    env.close()

    training_args = TrainingArguments(
        output_dir=f"./results_{env_name}_{trial}",
        num_train_epochs=training_epoch,  # Number of update epochs per rollout
        per_device_train_batch_size=batch_size,  # Batch size in trajectory units
        gradient_accumulation_steps=rollout * 2 // batch_size,
        learning_rate=3e-4,
        logging_steps=1,
        save_steps=1000,
        disable_tqdm=True,
        report_to="none",
        label_names=[
            "state",
            "action",
            "action_mask",
            "action_logit",
            "reward",
            "z_target",
            "intrinsic_reward",
            "mask",
        ],
        lr_scheduler_type="constant",
        dataloader_pin_memory=False,
    )

    trainer = RLTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=RLTrajectoryDataset([]),
        data_collator=trajectory_data_collator,
        callbacks=[LossLoggingCallback()],
    )
    trainer.create_optimizer()

    for epoch in range(training_iteration):
        print(f"\nEpoch {epoch}: Collecting trajectories on {env_name} ...")
        env_var_ = [
            EnvironmentVariable.build(ENV, model_mapping) for _ in range(rollout)
        ]
        trajectories = collect_trajectory(env_var_)
        # print(
        #     f"Env: {env_name} - Epoch {epoch} - Average Reward ({num_episodes} eps): {avg_reward}"
        # )
        dataset = RLTrajectoryDataset(trajectories)
        trainer.train_dataset = dataset
        trainer.first_batch = True
        print("Starting training ...")
        trainer.train()


if __name__ == "__main__":
    experiments = [
        dict(
            ENV=ChessEnv,
            config=RLConfig(
                hidden_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=64,
            ),
            batch_size=16,
            rollout=32,
            training_epoch=4,
            training_iteration=1000,
            trial=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    ]
    for opt in experiments:
        print("=" * 40)
        run_experiment(**opt)
