# Configs Reference

This page documents all configuration classes in TorchWM.

## DreamerConfig

Configuration for Dreamer agent training.

```python :class: thebe
@dataclass
class DreamerConfig:
    # Environment
    env_backend: str = "dmc"
    env: str = "walker-walk"
    env_instance: Optional[object] = None
    image_size: Tuple[int, int] = (64, 64)
    gym_render_mode: str = "rgb_array"

    # Unity ML-Agents
    unity_file_name: Optional[str] = None
    unity_behavior_name: Optional[str] = None
    unity_worker_id: int = 0
    unity_base_port: int = 5005
    unity_no_graphics: bool = True
    unity_time_scale: float = 20.0
    unity_quality_level: int = 1

    # Training
    algo: str = "Dreamerv1"
    exp_name: str = "lr1e-3"
    train: bool = True
    evaluate: bool = False
    seed: int = 1
    no_gpu: bool = False
    max_episode_length: int = 1000
    buffer_size: int = 800000
    time_limit: int = 1000

    # Model
    cnn_activation_function: str = "relu"
    dense_activation_function: str = "elu"
    obs_embed_size: int = 1024
    num_units: int = 400
    deter_size: int = 200
    stoch_size: int = 30
    action_repeat: int = 2
    action_noise: float = 0.3

    # Learning
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    total_steps: int = int(5e6)
    seed_steps: int = 5000
    update_steps: int = 100
    collect_steps: int = 1000
    batch_size: int = 50
    train_seq_len: int = 50
    imagine_horizon: int = 15

    # Loss
    free_nats: float = 3.0
    kl_scale: float = 1.0
    discount: float = 0.99
    td_lambda: float = 0.95
    kl_loss_coeff: float = 1.0
    kl_alpha: float = 0.8
    disc_loss_coeff: float = 10.0

    # Optimization
    adam_epsilon: float = 1e-7
    grad_clip_norm: float = 100.0

    # Exploration
    expl_amount: float = 0.3
    expl_decay: float = 0.0
    expl_min: float = 0.0

    # Logging
    log_every: int = 1e4
    log_scalars: bool = True
    log_images: bool = True
    log_videos: bool = False
    save_every: int = 1e5
    eval_every: int = 1e4
    eval_episodes: int = 10

    # WandB
    enable_wandb: bool = False
    wandb_api_key: str = ""
    wandb_project: str = "torchwm"
    wandb_entity: str = ""
    log_dir: str = "runs"

    # Operator parameters
    operator_image_size: int = 64
    operator_action_dim: int = 6
```

## JEPAConfig

Configuration for JEPA training.

```python :class: thebe
@dataclass
class JEPAConfig:
    # Meta
    use_bfloat16: bool = False
    model_name: str = "vit_base"
    load_checkpoint: bool = False
    read_checkpoint: Optional[str] = None
    copy_data: bool = False
    pred_depth: int = 6
    pred_emb_dim: int = 384

    # Data
    dataset: str = "imagenet"
    val_split: Optional[float] = None
    use_gaussian_blur: bool = True
    use_horizontal_flip: bool = True
    use_color_distortion: bool = True
    color_jitter_strength: float = 0.5
    batch_size: int = 64
    pin_mem: bool = True
    num_workers: int = 8
    root_path: str = os.environ.get("IMAGENET_ROOT", "/data/imagenet")
    image_folder: str = "train"
    crop_size: int = 224
    crop_scale: Tuple[float, float] = (0.67, 1.0)
    download: bool = False

    # Mask
    allow_overlap: bool = False
    patch_size: int = 16
    num_enc_masks: int = 1
    min_keep: int = 4
    enc_mask_scale: Tuple[float, float] = (0.15, 0.2)
    num_pred_masks: int = 1
    pred_mask_scale: Tuple[float, float] = (0.15, 0.2)
    aspect_ratio: Tuple[float, float] = (0.75, 1.5)

    # Optimization
    ema: Tuple[float, float] = (0.996, 1.0)
    clip_grad: float = 1.0
    weight_decay: float = 0.0
    epochs: int = 100
    warmup_epochs: int = 40
    start_epoch: int = 0
    lr: float = 1.5e-4
    min_lr: float = 1e-6
    accum_iter: int = 1
    output_dir: str = "./output"
    log_dir: str = "./output"
    device: str = "cuda"
    seed: int = 0
    resume: str = ""
    auto_resume: bool = True
    save_ckpt: bool = True
    save_ckpt_freq: int = 20
    save_ckpt_num: int = 3
    start_ckpt: str = ""
    debug: bool = False
    num_debug: int = 10
    wandb: bool = True
    wandb_project: str = "jepa"
    wandb_entity: str = ""
    enable_sweep: bool = False
    sweep_config: Dict[str, Any] = {}
    dist_eval: bool = True
    no_env: bool = False
    eval: bool = False
    disable_rel_pos_bias: bool = True
    disable_masking: bool = False
    enable_deepspeed: bool = False
    gradient_checkpointing: bool = True

    # Operator parameters
    operator_image_size: int = 224
    operator_patch_size: int = 16
    operator_mask_ratio: float = 0.75
```

## IRISConfig

Configuration for IRIS training.

```python :class: thebe
@dataclass
class IRISConfig:
    # Discrete Autoencoder (VQVAE)
    frame_height: int = 64
    frame_width: int = 64
    frame_channels: int = 3
    vocab_size: int = 512
    tokens_per_frame: int = 16
    token_embedding_dim: int = 512
    encoder_channels: int = 64
    encoder_layers: int = 4
    encoder_residual_blocks: int = 2
    decoder_depth: int = 32
    reconstruction_weight: float = 1.0
    commitment_weight: float = 0.25
    perceptual_weight: float = 1.0

    # Transformer (World Model)
    transformer_timesteps: int = 20
    transformer_embed_dim: int = 256
    transformer_layers: int = 10
    transformer_heads: int = 4
    transformer_dropout: float = 0.1

    # Actor-Critic
    imagination_horizon: int = 15
    discount: float = 0.99
    td_lambda: float = 0.9
    entropy_coef: float = 0.01
    actor_hidden_size: int = 512
    actor_layers: int = 4
    value_hidden_size: int = 512
    value_layers: int = 3

    # Training
    total_epochs: int = 600
    collection_epochs: int = 500
    env_steps_per_epoch: int = 200
    training_steps_per_epoch: int = 250
    model_learning_rate: float = 1e-4
    actor_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm: float = 10.0
    collect_epsilon: float = 0.1
    eval_temperature: float = 0.1
    start_autoencoder_after: int = 1
    start_transformer_after: int = 15
    start_actor_critic_after: int = 35
    autoencoder_batch_size: int = 256
    transformer_batch_size: int = 64
    actor_critic_batch_size: int = 64

    # Atari 100k Benchmark
    atari_100k: bool = True
    max_env_steps: int = 100000
    env_backend: str = "gym"
    env_name: str = "ALE/Pong-v5"
    action_repeat: int = 4
    max_episode_steps: int = 108000
    num_eval_episodes: int = 10
    eval_max_episode_steps: int = 27000
    buffer_capacity: int = 100000
    num_workers: int = 4
    prefetch_factor: int = 2

    # Logging
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    use_wandb: bool = True
    wandb_project: str = "iris"
    wandb_entity: str = ""

    # Operator parameters
    operator_seq_length: int = 512
    operator_vocab_size: int = 32000
```

## DiamondConfig

Configuration for Diamond (Diffusion + RL) training.

```python :class: thebe
@dataclass
class DiamondConfig:
    # Preset
    preset: Optional[str] = None  # "small", "medium", "large"

    # Environment
    game: str = "Breakout-v5"
    obs_size: int = 64
    frameskip: int = 4
    max_noop: int = 30
    terminate_on_life_loss: bool = True
    reward_clip: List[int] = field(default_factory=lambda: [-1, 0, 1])
    num_conditioning_frames: int = 4

    # Diffusion Model
    diffusion_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    diffusion_res_blocks: int = 2
    diffusion_cond_dim: int = 256
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: int = 7
    p_mean: float = -0.4
    p_std: float = 1.2
    sampling_method: str = "euler"
    num_sampling_steps: int = 3

    # Reward/Termination Model
    reward_channels: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    reward_res_blocks: int = 2
    reward_cond_dim: int = 128
    reward_lstm_dim: int = 512
    burn_in_length: int = 4

    # RL Agent
    actor_channels: List[int] = field(default_factory=lambda: [32, 32, 64, 64])
    actor_res_blocks: int = 1
    actor_lstm_dim: int = 512

    # Training
    num_epochs: int = 1000
    training_steps_per_epoch: int = 400
    batch_size: int = 32
    environment_steps_per_epoch: int = 100
    epsilon_greedy: float = 0.01
    imagination_horizon: int = 15
    discount_factor: float = 0.985
    entropy_weight: float = 0.001
    lambda_returns: float = 0.95
    learning_rate: float = 1e-4
    adam_epsilon: float = 1e-8
    weight_decay_diffusion: float = 1e-2
    weight_decay_reward: float = 1e-2
    weight_decay_actor: float = 0.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    num_seeds: int = 5
    seed: int = 0

    # Operator parameters
    operator_state_dim: int = 32
    operator_action_dim: int = 4
```

## Usage Patterns

### Basic Configuration

```python :class: thebe
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
cfg.total_steps = 1_000_000
```

### Environment-Specific Configs

```python :class: thebe
# DMC
cfg.env_backend = "dmc"
cfg.env = "walker-walk"

# Gym
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"

# Unity
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = "env.exe"
```

### Training Configs

```python :class: thebe
# Basic training
cfg.batch_size = 50
cfg.learning_rate = 6e-4
cfg.total_steps = 5_000_000

# Logging
cfg.enable_wandb = True
cfg.wandb_project = "my_project"

# Checkpointing
cfg.save_every = 100_000
```

### Advanced Configs

```python :class: thebe
# Custom model sizes
cfg.obs_embed_size = 2048
cfg.num_units = 600
cfg.deter_size = 300

# Exploration
cfg.expl_amount = 0.5
cfg.expl_decay = 0.0001

# Loss weights
cfg.kl_scale = 0.1
cfg.free_nats = 1.0
```