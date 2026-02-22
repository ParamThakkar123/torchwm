class DreamerConfig:
    """Configuration container for Dreamer training, evaluation, and environment setup.

    This class centralizes environment backend selection (DMC/Gym/Unity),
    model dimensions, replay and optimization settings, logging cadence, and
    checkpoint options consumed by `DreamerAgent`.
    """

    def __init__(self):
        # Environment selection.
        # dmc: DeepMind Control Suite
        # gym: generic Gym/Gymnasium env IDs or prebuilt env instances
        # unity_mlagents: Unity ML-Agents executable
        self.env_backend = "dmc"
        self.env = "walker-walk"
        self.env_instance = None
        self.image_size = (64, 64)
        self.gym_render_mode = "rgb_array"

        # Unity ML-Agents options.
        self.unity_file_name = None
        self.unity_behavior_name = None
        self.unity_worker_id = 0
        self.unity_base_port = 5005
        self.unity_no_graphics = True
        self.unity_time_scale = 20.0
        self.unity_quality_level = 1

        self.algo = "Dreamerv1"
        self.exp_name = "lr1e-3"
        self.train = True
        self.evaluate = False
        self.seed = 1
        self.no_gpu = False
        self.max_episode_length = 1000
        self.buffer_size = 800000
        self.time_limit = 1000
        self.cnn_activation_function = "relu"
        self.dense_activation_function = "elu"
        self.obs_embed_size = 1024
        self.num_units = 400
        self.deter_size = 200
        self.stoch_size = 30
        self.action_repeat = 2
        self.action_noise = 0.3
        self.total_steps = int(5e6)
        self.seed_steps = 5000
        self.update_steps = 100
        self.collect_steps = 1000
        self.batch_size = 50
        self.train_seq_len = 50
        self.imagine_horizon = 15
        self.use_disc_model = False
        self.free_nats = 3.0
        self.discount = 0.99
        self.td_lambda = 0.95
        self.kl_loss_coeff = 1.0
        self.kl_alpha = 0.8
        self.disc_loss_coeff = 10.0
        self.model_learning_rate = 6e-4
        self.actor_learning_rate = 8e-5
        self.value_learning_rate = 8e-5
        self.adam_epsilon = 1e-7
        self.grad_clip_norm = 100.0
        self.test = False
        self.test_interval = 10000
        self.test_episodes = 10
        self.scalar_freq = int(1e3)
        self.log_video_freq = -1
        self.max_videos_to_save = 2
        self.checkpoint_interval = 10000
        self.checkpoint_path = ""
        self.restore = False
        self.experience_replay = ""
        self.render = False
