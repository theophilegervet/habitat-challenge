from typing import Tuple
import gym
from gym.spaces import Dict as SpaceDict
from gym.spaces import Box, Discrete
import torch
import numpy as np
import torch.nn.functional as F

from .policy import Policy


class SemanticExplorationPolicy(Policy):
    """
    This predicts a high-level goal from map features.
    """

    def __init__(self, config):
        super().__init__(config)

        self._goal_update_steps = config.AGENT.POLICY.SEMANTIC.goal_update_steps
        self.inference_downscaling = config.AGENT.POLICY.SEMANTIC.inference_downscaling
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        self.local_map_size = (
            config.AGENT.SEMANTIC_MAP.map_size_cm //
            config.AGENT.SEMANTIC_MAP.global_downscaling //
            self.map_resolution //
            self.inference_downscaling
        )
        map_features_shape = (
            config.ENVIRONMENT.num_sem_categories + 8,
            self.local_map_size,
            self.local_map_size
        )

        # Only import ray if this class is instantiated
        from ray.rllib.agents import ppo
        from ray.rllib.models import ModelCatalog
        from .semantic_exploration_policy_network import SemanticExplorationPolicyModelWrapper
        from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper

        ModelCatalog.register_custom_model(
            "semexp_custom_model",
            SemanticExplorationPolicyModelWrapper
        )
        env_config = config.clone()
        env_config.defrost()
        env_config.NUM_ENVIRONMENTS = 1
        env_config.freeze()
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update({
            "env": SemanticExplorationPolicyInferenceEnv,
            #"env": SemanticExplorationPolicyTrainingEnvWrapper,
            "env_config": {"config": env_config},
            "model": {
                "custom_model": "semexp_custom_model",
                "custom_model_config": {
                    "map_features_shape": map_features_shape,
                    "hidden_size": 256,
                    "num_sem_categories": config.ENVIRONMENT.num_sem_categories,
                },
            },
            "framework": "torch",
            "normalize_actions": False,
            "_disable_preprocessor_api": True,
            "num_gpus_per_worker": 1,
        })
        algo = ppo.PPOTrainer(
            config=ppo_config,
            env=SemanticExplorationPolicyInferenceEnv
            #env=SemanticExplorationPolicyTrainingEnvWrapper
        )
        algo.restore(config.AGENT.POLICY.SEMANTIC.checkpoint_path)
        policy = algo.get_policy()
        self.dist_class = policy.dist_class
        self.model = policy.model

        # TODO How to load trained network weights from checkpoint without
        #  importing Ray?

    @property
    def goal_update_steps(self):
        return self._goal_update_steps

    def explore_otherwise(self,
                          map_features,
                          local_pose,
                          goal_category,
                          goal_map,
                          found_goal,
                          found_hint):
        batch_size, goal_map_size, _ = goal_map.shape
        map_features = F.avg_pool2d(map_features, self.inference_downscaling)
        obs = {"obs": {
            "map_features": map_features,
            "local_pose": local_pose,
            "goal_category": goal_category
        }}

        outputs, _ = self.model(obs)
        dist = self.dist_class(outputs, self.model)
        goal_action = dist.sample()
        goal_location = (torch.sigmoid(goal_action) * (goal_map_size - 1)).long()

        for e in range(batch_size):
            if not found_goal[e] and not found_hint[e]:
                goal_map[e, goal_location[e, 0], goal_location[e, 1]] = 1

        return goal_map


class SemanticExplorationPolicyInferenceEnv(gym.Env):
    """
    This environment is a lightweight environment to let us load
    a trained checkpoint of the semantic exploration policy without
    needing to instantiate the full training environment.

    This is needed because the full training environment needs access
    to a dataset that we don't have in the inference Docker.

    It would be great to not need this, but currently loading a
    checkpoint at inference time without introducing a dependency
    on Ray does not seem possible.
    """

    def __init__(self, rllib_config):
        config = rllib_config["config"]
        self.resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        self.global_map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.global_downscaling = config.AGENT.SEMANTIC_MAP.global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        global_map_size = self.global_map_size_cm // self.resolution
        self.global_h, self.global_w = global_map_size, global_map_size
        local_map_size = self.local_map_size_cm // self.resolution
        self.local_h, self.local_w = local_map_size, local_map_size
        self.inference_downscaling = config.AGENT.POLICY.SEMANTIC.inference_downscaling
        self.num_sem_categories = config.ENVIRONMENT.num_sem_categories

        self.observation_space = SpaceDict({
            "map_features": Box(
                low=0.,
                high=1.,
                shape=(
                    8 + self.num_sem_categories,
                    self.local_h // self.inference_downscaling,
                    self.local_w // self.inference_downscaling
                ),
                dtype=np.float32
            ),
            "local_pose": Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32,
            ),
            "goal_category": Discrete(6),
        })

        # Action space is logit pre sigmoid normalization to [0, 1]
        self.action_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )

    def _get_dummy_obs(self) -> dict:
        return {
            "map_features": np.zeros((
                8 + self.num_sem_categories,
                self.local_h // self.inference_downscaling,
                self.local_w // self.inference_downscaling
            ), dtype=np.float32),
            "local_pose": np.zeros(3, dtype=np.float32),
            "goal_category": 0
        }

    def reset(self) -> dict:
        obs = self._get_dummy_obs()
        print()
        print("__RESET__")
        for k, v in obs.items():
            print(k)
            try:
                print(v.shape)
                print(v.dtype)
            except:
                pass
            print(v)
        print()
        return obs

    # def step(self, goal_action: np.ndarray) -> Tuple[dict, float, bool, dict]:
    #     obs = self._get_dummy_obs()
    #     reward = 0.
    #     done = False
    #     info = {}
    #     return obs, reward, done, info
