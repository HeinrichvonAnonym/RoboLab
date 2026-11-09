"""LingBot-VLA v2 behind the existing reset/infer contract.

Official FeatureTransform apply, sample_actions, and unapply stay in
~/lingbot-vla-v2. This module only redirects the two hardcoded paths and
returns the absolute (chunk, 8) joint-plus-gripper command.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np


def lingbot_observation(observation):
    """Map this project's snapshot onto the Franka robot yaml origin keys."""
    return {
        "observation.images.kinec2": observation["image"]["kinec2"],
        "observation.images.realsense": observation["image"]["realsense"],
        "observation.state": np.asarray(observation["state"], dtype=np.float32),
        "task": observation["prompt"],
    }


def lingbot_action(result, chunk_size):
    if not isinstance(result, dict) or "action" not in result:
        keys = sorted(result) if isinstance(result, dict) else type(result).__name__
        raise ValueError(f"LingBot result is missing action, keys={keys}")
    action = np.asarray(result["action"])
    if action.shape != (chunk_size, 8) or not np.isfinite(action).all():
        raise ValueError(f"LingBot action must be finite {(chunk_size, 8)}, got {action.shape}")
    return {"action": np.asarray(action, dtype=np.float64)}


def _load_official(lingbot_root):
    path = Path(lingbot_root) / "deploy" / "lingbot_vla_v2_policy.py"
    spec = importlib.util.spec_from_file_location("robolab_lingbot_vla_v2_policy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _franka_server(official):
    class FrankaServer(official.LingbotVLAv2Server):
        """Same server, with this repo's training yaml and Franka robot yaml."""

        def __init__(self, training_config_path, robot_config_path, norm_stats_path, **kwargs):
            self._training_config_path = str(training_config_path)
            self._robot_config_path = str(robot_config_path)
            self._norm_stats_path = str(norm_stats_path)
            self._official = official
            super().__init__(**kwargs)

        def load_vla(self, path_to_pi_model):
            import yaml
            from types import SimpleNamespace

            mod = self._official
            print(f"loading model from: {path_to_pi_model}")
            with open(self._training_config_path, "r") as handle:
                training_config = yaml.safe_load(handle)

            training_model_config = training_config["model"]
            training_model_config.update(training_config["train"])
            config = mod.LingbotVLAV2Config(**training_model_config)
            for key, value in training_model_config.items():
                if not hasattr(config, key):
                    setattr(config, key, value)

            # Official deploy uses eager attention at evaluation time.
            config.attention_implementation = "eager"

            training_base_model = training_config["model"]["tokenizer_path"]
            if "qwen3" in training_base_model.lower() and "vl" in training_base_model.lower():
                model_name = "qwen3vl"
            else:
                raise ValueError(f"Unsupported base model of {path_to_pi_model}")
            base_model_path = os.environ.get("QWEN3VL_PATH", training_base_model) or mod.BASE_MODEL_PATH[model_name]
            config.tokenizer_path = base_model_path
            self.model_name = model_name

            self.config = config
            qwen_config = mod.AutoConfig.from_pretrained(base_model_path)
            self.merge_qwen_config(qwen_config)
            config = self.config

            if "vocab_size" in training_config["model"] and training_config["model"]["vocab_size"] != 0:
                config.vocab_size = training_config["model"]["vocab_size"]
            config.use_cache = True
            self.processor = mod.build_processor(base_model_path)
            self.language_tokenizer = self.processor.tokenizer
            data_config = SimpleNamespace(**training_config["data"])

            print("Initializing model ... ")
            self.vla = mod.LingBotVlaV2InferencePolicy(config, eval=True)
            self.load_model_weights(path_to_pi_model, strict=True)
            self.vla.feature_transform = None
            self.data_config = data_config
            self.config = config
            self.vla.model._use_compile_predict_velocity = bool(self.use_compile)
            self.vla.model._compiled_predict_velocity = None
            self.sample_actions_fn = self.vla.model.sample_actions
            if self.use_compile:
                import torch
                self.vla.model.qwenvl_with_expert = torch.compile(self.vla.model.qwenvl_with_expert)
                self.sample_actions_fn = torch.compile(self.vla.model.sample_actions)

            if self.robot_norm_path is None:
                self.robot_norm_path = data_config.norm_stats_file
            print("Model initialized ... ")
            return self.vla

        def reset(self, robo_name=None, path_to_pi_model=None):
            import torch
            import yaml
            mod = self._official
            if path_to_pi_model is not None:
                self.vla = self.load_vla(path_to_pi_model)
                if self.use_bf16:
                    self.vla = self.vla.to(torch.bfloat16).cuda().eval()
                else:
                    self.vla.model.float()
                    self.vla = self.vla.cuda().eval()

            self.global_step = 0
            self.last_action_chunk = None
            self.last_normalized_action_chunk = None
            self._rde_hist = []

            with open(self._robot_config_path, "r") as handle:
                self.robot_config = yaml.safe_load(handle)
            feature_transform = mod.FeatureTransform(
                self._robot_config_path,
                self.data_config,
                self.config,
                self.processor,
                chunk_size=self.config.chunk_size,
                norm_stats_path=self._norm_stats_path,
            )
            self.vla.feature_transform = feature_transform
            self.action_key = feature_transform.org_features["actions"]

    return FrankaServer


class LingbotPolicy:
    def __init__(self, cfg, kin):
        del kin
        os.environ["QWEN3VL_PATH"] = cfg["qwen3vl_path"]
        os.environ["RDE_N"] = "0"
        official = _load_official(cfg["lingbot_root"])
        self.chunk_size = cfg["chunk_size"]
        self.server = _franka_server(official)(
            training_config_path=cfg["lingbot_cli"],
            robot_config_path=cfg["lingbot_robot_config"],
            norm_stats_path=cfg["norm_stats"],
            path_to_pi_model=cfg["checkpoint_path"],
            robot_norm_path=cfg["norm_stats"],
            use_length=cfg["chunk_size"],
            chunk_ret=True,
            use_bf16=True,
            use_compile=False,
        )
        self.reset()

    def reset(self):
        self.server.reset()

    def infer(self, observation):
        return lingbot_action(self.server.infer(lingbot_observation(observation)), self.chunk_size)
