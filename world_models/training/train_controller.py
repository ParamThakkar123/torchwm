"""Training a linear controller on latent + recurrent state with CMA-ES.

This module provides functions to train a linear controller using Covariance
Matrix Adaptation Evolution Strategy (CMA-ES). The controller maps latent
and hidden states to actions for the learned world model.

Reference:
    Ha & Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution.
    https://arxiv.org/abs/1805.11111
"""

import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
import numpy as np
from tqdm import tqdm

from world_models.models.controller import Controller
from world_models.configs.wm_config import WMControllerConfig
from world_models.controller.rollout_generator import RolloutGenerator


def flatten_parameters(parameters):
    """Flatten model parameters into a single vector.

    Args:
        parameters: Iterator of model parameters.

    Returns:
        Flattened parameter vector as numpy array.
    """
    return np.concatenate([p.data.cpu().numpy().flatten() for p in parameters])


def load_parameters(params, controller):
    """Load flattened parameters into controller model.

    Args:
        params: Flattened parameter vector as numpy array.
        controller: Controller model to load parameters into.
    """
    pointer = 0
    for param in controller.parameters():
        param_shape = param.shape
        size = 1
        for s in param_shape:
            size *= s
        param.data.copy_(
            torch.from_numpy(params[pointer : pointer + size]).view(param_shape)
        )
        pointer += size


def slave_routine(p_queue, r_queue, e_queue, p_index, config, time_limit):
    """Worker thread routine for parallel rollout evaluation.

    Threads interact with p_queue (parameters), r_queue (results), and e_queue (end signal).
    They pull parameters from p_queue, execute rollouts, then place results in r_queue.

    Args:
        p_queue: Queue containing (s_id, parameters) to evaluate.
        r_queue: Queue where to place results (s_id, reward).
        e_queue: End queue - when non-empty, thread terminates.
        p_index: Process index for GPU assignment.
        config: Controller configuration.
        time_limit: Maximum steps per episode.
    """
    gpu = p_index % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    sys.stdout = open(join(config.logdir, "tmp", str(getpid()) + ".out"), "a")
    sys.stderr = open(join(config.logdir, "tmp", str(getpid()) + ".err"), "a")

    with torch.no_grad():
        r_gen = RolloutGenerator(config.logdir, device, time_limit=time_limit)

        while e_queue.empty():
            if p_queue.empty():
                sleep(0.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


def evaluate(solutions, results, rollouts, p_queue, r_queue):
    """Evaluate current controller.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    Args:
        solutions: CMA set of solutions.
        results: Corresponding results.
        rollouts: Number of rollouts for evaluation.

    Returns:
        Tuple of (best_params, mean_reward, std_reward).
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(0.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)


def train_controller(config: WMControllerConfig) -> None:
    """Train a linear controller using CMA-ES.

    This function trains a controller that maps latent and hidden states to actions.
    It uses parallel workers to evaluate candidate controllers and CMA-ES for optimization.

    Args:
        config: WMControllerConfig containing training hyperparameters.

    The training process includes:
        - Setting up parallel evaluation workers
        - Running CMA-ES optimization
        - Evaluating and saving best controller

    Example:
        >>> config = WMControllerConfig({
        ...     'latent_size': 32,
        ...     'hidden_size': 200,
        ...     'action_size': 3,
        ...     'logdir': 'results',
        ...     'pop_size': 10,
        ...     'n_samples': 4,
        ... })
        >>> train_controller(config)
    """
    n_samples = config.n_samples
    pop_size = config.pop_size
    num_workers = min(config.max_workers, n_samples * pop_size)
    time_limit = config.time_limit

    tmp_dir = join(config.logdir, "tmp")
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))

    ctrl_dir = join(config.logdir, "ctrl")
    if not exists(ctrl_dir):
        mkdir(ctrl_dir)

    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    for p_index in range(num_workers):
        Process(
            target=slave_routine,
            args=(p_queue, r_queue, e_queue, p_index, config, time_limit),
        ).start()

    controller = Controller(config.latent_size, config.hidden_size, config.action_size)

    cur_best = None
    ctrl_file = join(ctrl_dir, "best.tar")
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={"cuda:0": "cpu"})
        cur_best = -state["reward"]
        controller.load_state_dict(state["state_dict"])
        print(f"Previous best was {-cur_best}...")

    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(
        flatten_parameters(parameters), 0.1, {"popsize": pop_size}
    )

    epoch = 0
    log_step = 3
    while not es.stop():
        if cur_best is not None and -cur_best > config.target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size
        solutions = es.ask()

        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        if config.display:
            pbar = tqdm(total=pop_size * n_samples)
        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                sleep(0.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples
            if config.display:
                pbar.update(1)
        if config.display:
            pbar.close()

        es.tell(solutions, r_list)
        es.disp()

        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(
                solutions, r_list, n_samples * pop_size, p_queue, r_queue
            )
            print(f"Current evaluation: {best}")
            if not cur_best or cur_best > best:
                cur_best = best
                print(f"Saving new best with value {-cur_best} +- {std_best}...")
                load_parameters(best_params, controller)
                torch.save(
                    {
                        "epoch": epoch,
                        "reward": -cur_best,
                        "state_dict": controller.state_dict(),
                    },
                    join(ctrl_dir, "best.tar"),
                )
            if -best > config.target_return:
                print(f"Terminating controller training with value {best}...")
                break

        epoch += 1

    es.result_pretty()
    e_queue.put("EOP")
