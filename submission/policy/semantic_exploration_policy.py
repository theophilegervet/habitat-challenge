import torch
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
        env_config = config.copy()
        env_config.defrost()
        env_config.NUM_ENVIRONMENTS = 1
        env_config.freeze()
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update({
            "env": SemanticExplorationPolicyTrainingEnvWrapper,
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
            env=SemanticExplorationPolicyTrainingEnvWrapper
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
