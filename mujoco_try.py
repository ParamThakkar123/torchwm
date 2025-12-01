from world_models.models.planet import Planet
from envs.mujoco_env import make_half_cheetah_env

env = make_half_cheetah_env(
    version="v4",
    forward_reward_weight=0.1,
    reset_noise_scale=0.1,
    render_mode="rgb_array",
)

planet = Planet(
    env=env,
    bit_depth=10,
    device=None,
    state_size=200,
    latent_size=30,
    embedding_size=1024,
    memory_size=100,
    policy_cfg={
        "planning_horizon": 30,
        "num_candidates": 2000,
        "num_iterations": 15,
        "top_candidates": 200,
    },
    headless=True,
    max_episode_steps=1000,
    action_repeats=1,
    results_dir="results/halfcheetah_planet",
)

planet.warmup(n_episodes=5, random_policy=True)

planet.train(
    epochs=100, steps_per_epoch=150, batch_size=32, H=50, beta=1.0, save_every=25
)
