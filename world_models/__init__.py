"""
TorchWM - Modular PyTorch Library for World Models

Public API exports for easy imports. All model components are exposed via this
simple interface for use in building custom models.

Usage:
    # Main agents
    from world_models import DreamerAgent, DreamerConfig
    agent = DreamerAgent(DreamerConfig())

    # Building blocks for custom models
    from world_models import ConvEncoder, ConvDecoder, RSSM, ReplayBuffer
    encoder = ConvEncoder(input_shape=(3, 64, 64), embed_size=256, activation='relu')
"""

__version__ = "0.4.0"


def __getattr__(name):
    """Lazy import to avoid loading unused modules and their dependencies."""

    # =====================================================================
    # AGENTS & HIGH-LEVEL MODELS
    # =====================================================================
    if name in (
        "Dreamer",
        "DreamerAgent",
        "Planet",
        "JEPAAgent",
        "VisionTransformer",
        "ModularRSSM",
        "create_modular_rssm",
        # Genie world model components
        "Genie",
        "LatentActionModel",
        "DynamicsModel",
        "create_genie",
        "create_genie_small",
        "create_genie_large",
        "create_latent_action_model",
        "create_dynamics_model",
    ):
        from world_models import models as _models

        return getattr(_models, name)

    # =====================================================================
    # RSSM & STATE SPACE MODELS
    # =====================================================================
    if name in (
        "RSSM",
        "RecurrentStateSpaceModel",
    ):
        from world_models import models as _models

        return getattr(_models, name)

    # =====================================================================
    # VISION: ENCODERS & DECODERS
    # =====================================================================
    if name in (
        "ConvEncoder",
        "CNNEncoder",
        "ConvDecoder",
        "CNNDecoder",
        "DenseDecoder",
        "ActionDecoder",
        "TanhBijector",
        "SampleDist",
        "IRISEncoder",
        "IRISDecoder",
    ):
        from world_models import vision as _vision

        return getattr(_vision, name)

    # =====================================================================
    # VISION: VIDEO TOKENIZER & VQ
    # =====================================================================
    if name in (
        "VideoTokenizer",
        "create_video_tokenizer",
        "VectorQuantizer",
        "VectorQuantizerEMA",
    ):
        from world_models import vision as _vision

        return getattr(_vision, name)

    # =====================================================================
    # MEMORY & REPLAY BUFFERS
    # =====================================================================
    if name in (
        "ReplayBuffer",
        "Memory",
        "Episode",
        "IRISReplayBuffer",
        "IRISOnPolicyBuffer",
    ):
        from world_models import memory as _memory

        return getattr(_memory, name)

    # =====================================================================
    # DIFFUSION MODELS
    # =====================================================================
    if name in (
        "DiT",
        "PatchEmbed",
        "PatchUnEmbed",
        "DDPM",
        "ActorCriticNetwork",
        "RewardTerminationModel",
        "sinusoidal_time_embedding",
    ):
        from world_models.models import diffusion as _diffusion

        return getattr(_diffusion, name)

    # =====================================================================
    # TRANSFORMER BLOCKS & LAYERS
    # =====================================================================
    if name in (
        "STTransformer",
        "MultiHeadSelfAttention",
        "MultiHeadAttention",
        "AdaLNNormalization",
        "RMSNorm",
    ):
        from world_models import blocks as _blocks

        return getattr(_blocks, name)

    # =====================================================================
    # CONTROLLERS & POLICIES
    # =====================================================================
    if name in (
        "RSSMPolicy",
        "RolloutGenerator",
        "IRISActor",
        "IRISCritic",
        "IRISPolicy",
        "CNNFeatureExtractor",
    ):
        from world_models import controller as _controller

        return getattr(_controller, name)

    # =====================================================================
    # CONFIGS
    # =====================================================================
    if name in (
        "DreamerConfig",
        "JEPAConfig",
        "DiTConfig",
        "get_dit_config",
        "DiamondConfig",
        "IRISConfig",
        "GenieConfig",
        "GenieSmallConfig",
        "STTransformerConfig",
        "VideoTokenizerConfig",
        "LatentActionModelConfig",
        "DynamicsModelConfig",
        "ATARI_100K_GAMES",
        "HUMAN_SCORES",
        "RANDOM_SCORES",
        # Genie configs
        "GenieConfig",
        "GenieSmallConfig",
        "STTransformerConfig",
        "VideoTokenizerConfig",
        "LatentActionModelConfig",
        "DynamicsModelConfig",
    ):
        from world_models import configs as _configs

        return getattr(_configs, name)

    # =====================================================================
    # ENVIRONMENTS
    # =====================================================================
    if name in (
        "make_atari_env",
        "list_available_atari_envs",
        "make_atari_vector_env",
        "make_humanoid_env",
        "make_half_cheetah_env",
        "GymImageEnv",
        "make_gym_env",
        "DeepMindControlEnv",
        "UnityMLAgentsEnv",
        "make_unity_mlagents_env",
        "MujocoEnv",
        "TimeLimit",
        "ActionRepeat",
        "NormalizeActions",
        "ObsDict",
        "OneHotAction",
        "RewardObs",
        "ResizeImage",
        "RenderImage",
        "SelectAction",
    ):
        from world_models import envs as _envs

        return getattr(_envs, name)

    # =====================================================================
    # INFERENCE OPERATORS
    # =====================================================================
    if name in (
        "OperatorABC",
        "DreamerOperator",
        "JEPAOperator",
        "IrisOperator",
        "PlaNetOperator",
        "get_operator",
    ):
        from world_models.inference import operators as _ops

        return getattr(_ops, name)

    # =====================================================================
    # REWARD & VALUE MODELS
    # =====================================================================
    if name in (
        "RewardModel",
        "ValueModel",
        "DreamerRewardModel",
        "DreamerValueModel",
    ):
        from world_models import reward as _reward

        return getattr(_reward, name)

    # Device utilities
    if name in (
        "get_device",
        "get_device_info",
        "set_device_for_model",
        "supports_bfloat16",
        "supports_mixed_precision",
    ):
        from world_models import device as _device

        return getattr(_device, name)

    # =====================================================================
    # LATENT ACTION & DYNAMICS MODELS (Genie sub-components)
    # =====================================================================
    if name in (
        "LatentActionModel",
        "DynamicsModel",
        "create_latent_action_model",
        "create_dynamics_model",
    ):
        from world_models import models as _models

        return getattr(_models, name)

    # =====================================================================
    # UTILITIES
    # =====================================================================
    if name in (
        "Logger",
        "FreezeParameters",
        "compute_return",
        "preprocess_obs",
    ):
        from world_models import utils as _utils

        return getattr(_utils, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =====================================================================
# PUBLIC API - All exports with descriptions
# =====================================================================
__all__ = [
    # ---------------------------------------------------------------------
    # VERSION
    # ---------------------------------------------------------------------
    "__version__",
    # ---------------------------------------------------------------------
    # AGENTS (High-level training wrappers)
    # ---------------------------------------------------------------------
    "DreamerAgent",  # High-level Dreamer training API
    "JEPAAgent",  # JEPA agent for self-supervised learning
    "Planet",  # PlaNet planning agent
    "VisionTransformer",  # Vision Transformer (ViT) for image encoding
    "ModularRSSM",  # Modular RSSM with swappable components
    "create_modular_rssm",  # Factory for ModularRSSM
    # ---------------------------------------------------------------------
    # CORE WORLD MODELS
    # ---------------------------------------------------------------------
    "Dreamer",  # Core Dreamer implementation with RSSM, actor, critic
    "Genie",  # Generative Interactive Environment model
    "create_genie",  # Factory function for Genie
    "create_genie_small",  # Small variant of Genie
    "create_genie_large",  # Large variant of Genie
    # ---------------------------------------------------------------------
    # STATE SPACE MODELS
    # ---------------------------------------------------------------------
    "RSSM",  # Recurrent State-Space Model (Dreamer-style)
    "RecurrentStateSpaceModel",  # Alternative name for RSSM
    "DreamerRSSM",  # Dreamer-specific RSSM implementation
    # ---------------------------------------------------------------------
    # VISION COMPONENTS
    # ---------------------------------------------------------------------
    "ConvEncoder",  # Convolutional encoder (Dreamer)
    "CNNEncoder",  # CNN encoder (PlaNet)
    "ConvDecoder",  # Convolutional decoder (Dreamer)
    "CNNDecoder",  # CNN decoder (PlaNet)
    "DenseDecoder",  # MLP decoder for rewards/values
    "ActionDecoder",  # Dreamer policy head
    "TanhBijector",  # Action squashing transformation
    "SampleDist",  # Distribution wrapper for sampling
    "IRISEncoder",  # IRIS-specific encoder
    "IRISDecoder",  # IRIS-specific decoder
    # ---------------------------------------------------------------------
    # VIDEO TOKENIZER
    # ---------------------------------------------------------------------
    "VideoTokenizer",  # VQ-VAE video tokenizer for Genie
    "create_video_tokenizer",  # Factory for VideoTokenizer
    "VectorQuantizer",  # Basic vector quantization
    "VectorQuantizerEMA",  # EMA-based vector quantization
    # ---------------------------------------------------------------------
    # MEMORY SYSTEMS
    # ---------------------------------------------------------------------
    "ReplayBuffer",  # Experience replay buffer (Dreamer)
    "Memory",  # Base memory class (deque-based)
    "Episode",  # Episode memory storage
    "IRISReplayBuffer",  # IRIS replay buffer
    "IRISOnPolicyBuffer",  # IRIS on-policy buffer
    # ---------------------------------------------------------------------
    # DIFFUSION MODELS
    # ---------------------------------------------------------------------
    "DiT",  # Diffusion Transformer
    "PatchEmbed",  # Image patch embedding for DiT
    "PatchUnEmbed",  # Patch unembedding (decode tokens to image)
    "DDPM",  # Denoising Diffusion Probabilistic Model
    "ActorCriticNetwork",  # Diffusion-based actor-critic
    "RewardTerminationModel",  # Reward/termination prediction head
    "sinusoidal_time_embedding",  # Time embedding for diffusion
    # ---------------------------------------------------------------------
    # TRANSFORMER BLOCKS
    # ---------------------------------------------------------------------
    "STTransformer",  # Spatiotemporal Transformer
    "MultiHeadSelfAttention",  # Self-attention module
    "MultiHeadAttention",  # Multi-head attention (alias)
    "AdaLNNormalization",  # Adaptive Layer Normalization
    "RMSNorm",  # Root Mean Square Normalization
    # ---------------------------------------------------------------------
    # CONTROLLERS & POLICIES
    # ---------------------------------------------------------------------
    "RSSMPolicy",  # RSSM-based policy
    "RolloutGenerator",  # Trajectory rollout generator
    "IRISActor",  # IRIS actor network
    "IRISCritic",  # IRIS critic network
    "IRISPolicy",  # IRIS planning policy
    "CNNFeatureExtractor",  # CNN feature extractor for IRIS
    # ---------------------------------------------------------------------
    # GENIE SUB-COMPONENTS
    # ---------------------------------------------------------------------
    "LatentActionModel",  # Latent action learning model
    "DynamicsModel",  # Future frame prediction model
    "create_latent_action_model",  # Factory for LatentActionModel
    "create_dynamics_model",  # Factory for DynamicsModel
    # ---------------------------------------------------------------------
    # CONFIGS
    # ---------------------------------------------------------------------
    "DreamerConfig",
    "JEPAConfig",
    "DiTConfig",
    "get_dit_config",
    "DiamondConfig",
    "IRISConfig",
    "GenieConfig",
    "GenieSmallConfig",
    "STTransformerConfig",
    "VideoTokenizerConfig",
    "LatentActionModelConfig",
    "DynamicsModelConfig",
    "ATARI_100K_GAMES",
    "HUMAN_SCORES",
    "RANDOM_SCORES",
    # ---------------------------------------------------------------------
    # ENVIRONMENTS
    # ---------------------------------------------------------------------
    "make_atari_env",
    "list_available_atari_envs",
    "make_atari_vector_env",
    "make_humanoid_env",
    "make_half_cheetah_env",
    "make_gym_env",
    "GymImageEnv",
    "DeepMindControlEnv",
    "UnityMLAgentsEnv",
    "make_unity_mlagents_env",
    "MujocoEnv",
    "TimeLimit",
    "ActionRepeat",
    "NormalizeActions",
    "ObsDict",
    "OneHotAction",
    "RewardObs",
    "ResizeImage",
    "RenderImage",
    "SelectAction",
    # ---------------------------------------------------------------------
    # INFERENCE OPERATORS
    # ---------------------------------------------------------------------
    "OperatorABC",
    "DreamerOperator",
    "JEPAOperator",
    "IrisOperator",
    "PlaNetOperator",
    "get_operator",
    # ---------------------------------------------------------------------
    # REWARD & VALUE MODELS
    # ---------------------------------------------------------------------
    "RewardModel",
    "ValueModel",
    "DreamerRewardModel",
    "DreamerValueModel",
    # ---------------------------------------------------------------------
    # Device utilities
    # ---------------------------------------------------------------------
    "get_device",
    "get_device_info",
    "set_device_for_model",
    "supports_bfloat16",
    "supports_mixed_precision",
    # ---------------------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------------------
    "Logger",
    "FreezeParameters",
    "compute_return",
    "preprocess_obs",
]
