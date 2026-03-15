class IRISConfig:
    """Configuration for IRIS (Imagination with auto-Regression over an Inner Speech)

    Based on paper: "Transformers are Sample-Efficient World Models"
    Implements discrete autoencoder + autoregressive Transformer for sample-efficient RL.
    """

    def __init__(self):
        # === Discrete Autoencoder (VQVAE) ===
        self.frame_height = 64
        self.frame_width = 64
        self.frame_channels = 3

        self.vocab_size = 512  # N: vocabulary size
        self.tokens_per_frame = 16  # K: tokens per frame
        self.token_embedding_dim = 512  # d: embedding dimension

        # Encoder settings
        self.encoder_channels = 64
        self.encoder_layers = 4
        self.encoder_residual_blocks = 2

        # Decoder settings
        self.decoder_depth = 32

        # Loss weights
        self.reconstruction_weight = 1.0
        self.commitment_weight = 0.25  # From VQGAN paper
        self.perceptual_weight = 1.0

        # === Transformer (World Model) ===
        self.transformer_timesteps = 20  # L: sequence length
        self.transformer_embed_dim = 256  # D
        self.transformer_layers = 10  # M
        self.transformer_heads = 4
        self.transformer_dropout = 0.1

        # === Actor-Critic ===
        self.imagination_horizon = 20  # H
        self.discount = 0.99  # gamma
        self.td_lambda = 0.9  # lambda for lambda-return
        self.entropy_coef = 0.01  # eta: entropy coefficient

        # Policy network
        self.actor_hidden_size = 512
        self.actor_layers = 4

        # Value network
        self.value_hidden_size = 512
        self.value_layers = 3

        # === Training ===
        self.total_epochs = 600
        self.collection_epochs = 500
        self.env_steps_per_epoch = 400
        self.training_steps_per_epoch = 500

        # Optimization
        self.model_learning_rate = 1e-4
        self.actor_learning_rate = 1e-4
        self.value_learning_rate = 1e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.weight_decay = 0.01
        self.grad_clip_norm = 10.0

        # Exploration
        self.collect_epsilon = 0.01
        self.eval_temperature = 0.1

        # Warm-start delays (epochs)
        self.start_autoencoder_after = 1
        self.start_transformer_after = 10
        self.start_actor_critic_after = 25

        # Batch sizes
        self.autoencoder_batch_size = 256
        self.transformer_batch_size = 64
        self.actor_critic_batch_size = 64

        # === Atari 100k Benchmark ===
        self.atari_100k = True
        self.max_env_steps = 100000  # ~2 hours of gameplay

        # === Environment ===
        self.env_backend = "gym"
        self.env = "ALE/Pong-v5"
        self.action_repeat = 4

        # === Logging ===
        self.log_interval = 1000
        self.eval_episodes = 100
        self.checkpoint_interval = 50

    def get_frame_shape(self):
        return (self.frame_channels, self.frame_height, self.frame_width)

    def get_autoencoder_config(self):
        return {
            "vocab_size": self.vocab_size,
            "tokens_per_frame": self.tokens_per_frame,
            "embedding_dim": self.token_embedding_dim,
            "encoder_channels": self.encoder_channels,
            "encoder_layers": self.encoder_layers,
            "encoder_residual_blocks": self.encoder_residual_blocks,
            "decoder_depth": self.decoder_depth,
            "frame_shape": self.get_frame_shape(),
            "reconstruction_weight": self.reconstruction_weight,
            "commitment_weight": self.commitment_weight,
            "perceptual_weight": self.perceptual_weight,
        }

    def get_transformer_config(self):
        return {
            "timesteps": self.transformer_timesteps,
            "embed_dim": self.transformer_embed_dim,
            "layers": self.transformer_layers,
            "heads": self.transformer_heads,
            "dropout": self.transformer_dropout,
            "vocab_size": self.vocab_size,
            "tokens_per_frame": self.tokens_per_frame,
            "action_size": None,  # Set at runtime
        }

    def get_rl_config(self):
        return {
            "imagination_horizon": self.imagination_horizon,
            "discount": self.discount,
            "td_lambda": self.td_lambda,
            "entropy_coef": self.entropy_coef,
            "actor_hidden_size": self.actor_hidden_size,
            "actor_layers": self.actor_layers,
            "value_hidden_size": self.value_hidden_size,
            "value_layers": self.value_layers,
            "frame_shape": self.get_frame_shape(),
        }
