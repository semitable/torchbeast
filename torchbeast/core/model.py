import numpy as np
import torch
from torch import nn
from typing import List
from gym.spaces import flatdim
import torch.nn.functional as F

from einops import rearrange

class MultiAgentFCNetwork(nn.Module):
    def _init_layer(self, m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None):
        mods = []

        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(self._init_layer(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        return nn.Sequential(*mods)

    def __init__(self, input_sizes, idims, output_sizes):
        super().__init__()

        n_agents = len(input_sizes)
        self.models = nn.ModuleList()

        for in_size, out_size in zip(input_sizes, output_sizes):
            dims = [in_size] + idims + [out_size]
            self.models.append(self._make_fc(dims))

    def forward(self, inputs: List[torch.Tensor]):
        futures = [
            torch.jit.fork(model, inputs[i]) for i, model in enumerate(self.models)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        return results



class FCNet(nn.Module):
    def __init__(self, observation_space, action_space, use_lstm=False):
        super(FCNet, self).__init__()

        self.n_agents = len(observation_space)
        self.observation_shape = observation_space
        self.action_space = action_space
        self.use_lstm = False

        obs_shape = [flatdim(o) for o in observation_space]
        action_shape = [flatdim(a) for a in action_space]

        self.policy = MultiAgentFCNetwork(
            obs_shape, [64, 64], action_shape
        )

        self.baseline = MultiAgentFCNetwork(
            obs_shape, [64, 64], self.n_agents*[1]
        )

        # core_output_size = 64

        # self.use_lstm = use_lstm
        # if use_lstm:
        #     self.core = nn.LSTM(core_output_size, core_output_size, num_layers=1)

        # self.policy = nn.Linear(core_output_size, self.num_actions)
        # self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):

        inputs = inputs["obs"]

        batch_size = inputs.shape[1]
        
        inputs = torch.split(inputs, 1, 2)
        inputs = [rearrange(i, "T B 1 F -> T B F") for i in inputs]


        policy_logits = self.policy(inputs)
        baseline = self.baseline(inputs)

        action = [torch.multinomial(F.softmax(rearrange(p, "T B F -> (T B) F"), dim=-1), num_samples=1) for p in policy_logits]
        action = [rearrange(a, "(T B) 1 -> T B 1", B=batch_size) for a in action]

        return (
            (action, policy_logits, baseline),
            core_state,
        )
        # x = inputs["frame"]  # [T, B, C, H, W].
        # T, B, *_ = x.shape
        # x = torch.flatten(x, 0, 1)  # Merge time and batch.
        # x = x.float() / 255.0
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = x.view(T * B, -1)
        # x = F.relu(self.fc(x))

        # one_hot_last_action = F.one_hot(
        #     inputs["last_action"].view(T * B), self.num_actions
        # ).float()
        # clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        # core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        # if self.use_lstm:
        #     core_input = core_input.view(T, B, -1)
        #     core_output_list = []
        #     notdone = (~inputs["done"]).float()
        #     for input, nd in zip(core_input.unbind(), notdone.unbind()):
        #         # Reset core state to zero whenever an episode ended.
        #         # Make `done` broadcastable with (num_layers, B, hidden_size)
        #         # states:
        #         nd = nd.view(1, -1, 1)
        #         core_state = tuple(nd * s for s in core_state)
        #         output, core_state = self.core(input.unsqueeze(0), core_state)
        #         core_output_list.append(output)
        #     core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        # else:
        #     core_output = core_input
        #     core_state = tuple()

        # policy_logits = self.policy(core_output)
        # baseline = self.baseline(core_output)

        # if self.training:
        #     action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        # else:
        #     # Don't sample when testing.
        #     action = torch.argmax(policy_logits, dim=1)

        # policy_logits = policy_logits.view(T, B, self.num_actions)
        # baseline = baseline.view(T, B)
        # action = action.view(T, B)

        # return (
        #     dict(policy_logits=policy_logits, baseline=baseline, action=action),
        #     core_state,
        # )