"""
Hybrid CNN + MLP feature extractor for the assembly-RL Dict observation space
(image + proprio + force_torque + ee_pose), used by both SAC and PPO so the
two algorithms are compared on identical perception/state encoding (see
train.py's --algo flag) -- the only thing that should differ between the two
runs is the RL algorithm itself.

Architecture:
    image (84x84x3 uint8)        -> small conv stack -> 128-d
    proprio + force_torque + ee_pose (25-d, concatenated) -> 2-layer MLP -> 64-d
    concat(128, 64) -> Linear -> features_dim (default 256)

This is intentionally a lighter conv stack than the original DQN "NatureCNN"
(three conv layers is overkill for an 84x84 close-up wrist view of a
peg/hole, and a smaller network trains faster -- relevant for a sample-
efficiency-focused comparison against SAC's off-policy replay).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AssemblyFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,
                 cnn_out: int = 128, state_out: int = 64):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        # envs/assembly_env.py's renderer produces HWC, but by the time this
        # extractor is constructed the observation_space reflects whatever
        # rl/train.py's VecEnv stack does to it -- VecTransposeImage (applied
        # there) transposes to CHW, so image_space.shape is (C, H, W) here.
        # Detect the channel axis defensively instead of assuming a fixed layout.
        shape = image_space.shape
        channel_axis = int(np.argmin(shape))  # the channel dim (1 or 3) is always the smallest
        n_input_channels = shape[channel_axis]
        spatial = tuple(s for i, s in enumerate(shape) if i != channel_axis)
        self._channel_axis = channel_axis

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, *spatial)
            cnn_flat_dim = self.cnn(sample).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(cnn_flat_dim, cnn_out), nn.ReLU())

        state_dim = (observation_space["proprio"].shape[0]
                     + observation_space["force_torque"].shape[0]
                     + observation_space["ee_pose"].shape[0])
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, state_out), nn.ReLU(),
        )

        self.joint = nn.Sequential(nn.Linear(cnn_out + state_out, features_dim), nn.ReLU())

    def forward(self, obs: dict) -> torch.Tensor:
        # NOTE: obs["image"] arrives already NCHW here, not the NHWC that
        # envs/assembly_env.py's renderer produces -- SB3's VecTransposeImage
        # (applied in rl/train.py's build_vec_env) does that transpose at the
        # VecEnv level before observations ever reach the policy/extractor.
        image = obs["image"].float() / 255.0
        img_feat = self.cnn_head(self.cnn(image))

        state = torch.cat([obs["proprio"], obs["force_torque"], obs["ee_pose"]], dim=-1)
        state_feat = self.state_mlp(state)

        return self.joint(torch.cat([img_feat, state_feat], dim=-1))


POLICY_KWARGS = dict(
    features_extractor_class=AssemblyFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256, cnn_out=128, state_out=64),
    net_arch=[256, 256],
    # SB3's preprocess_obs() auto-/255-normalizes any Box subspace it detects as an
    # image *before* the features extractor ever sees it; our extractor does that
    # division itself (module docstring), so this is required to avoid normalizing
    # twice (which silently crushes pixel values toward 0 and cripples the CNN).
    normalize_images=False,
)
