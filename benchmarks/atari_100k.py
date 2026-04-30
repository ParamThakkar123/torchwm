"""
IRIS Atari 100k Benchmark Runner

Runs IRIS on all 26 Atari games from the Atari 100k benchmark.
Computes human-normalized scores and compares to baselines.

Based on paper: "Transformers are Sample-Efficient World Models"
"""

from world_models.training.train_iris import IRISTrainer
from world_models.configs.iris_config import IRISConfig
import numpy as np
import json
import os
from typing import List, Dict


# Atari 100k benchmark games
ATARI_100K_GAMES = [
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
    "ALE/Assault-v5",
    "ALE/Asterix-v5",
    "ALE/BankHeist-v5",
    "ALE/BattleZone-v5",
    "ALE/Boxing-v5",
    "ALE/Breakout-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/DemonAttack-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Hero-v5",
    "ALE/Jamesbond-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MsPacman-v5",
    "ALE/Pong-v5",
    "ALE/PrivateEye-v5",
    "ALE/Qbert-v5",
    "ALE/RoadRunner-v5",
    "ALE/Seaquest-v5",
    "ALE/UpNDown-v5",
]


# Human scores for computing human-normalized score
# These are from the paper (Wang et al., 2016)
HUMAN_SCORES = {
    "Alien": 7127.7,
    "Amidar": 1719.5,
    "Assault": 742.0,
    "Asterix": 8503.3,
    "BankHeist": 753.1,
    "BattleZone": 37187.5,
    "Boxing": 12.1,
    "Breakout": 30.5,
    "ChopperCommand": 7387.8,
    "CrazyClimber": 35829.4,
    "DemonAttack": 1971.0,
    "Freeway": 29.6,
    "Frostbite": 4334.7,
    "Gopher": 2412.5,
    "Hero": 30826.4,
    "Jamesbond": 302.8,
    "Kangaroo": 3035.0,
    "Krull": 2665.5,
    "KungFuMaster": 22736.3,
    "MsPacman": 6951.6,
    "Pong": 14.6,
    "PrivateEye": 69571.3,
    "Qbert": 13455.0,
    "RoadRunner": 7845.0,
    "Seaquest": 42054.7,
    "UpNDown": 11693.2,
}


# Random baseline scores
RANDOM_SCORES = {
    "Alien": 227.8,
    "Amidar": 5.8,
    "Assault": 222.4,
    "Asterix": 210.0,
    "BankHeist": 14.2,
    "BattleZone": 2360.0,
    "Boxing": 0.1,
    "Breakout": 1.7,
    "ChopperCommand": 811.0,
    "CrazyClimber": 10780.5,
    "DemonAttack": 152.1,
    "Freeway": 0.0,
    "Frostbite": 65.2,
    "Gopher": 257.6,
    "Hero": 1027.0,
    "Jamesbond": 29.0,
    "Kangaroo": 52.0,
    "Krull": 1598.0,
    "KungFuMaster": 258.5,
    "MsPacman": 307.3,
    "Pong": -20.7,
    "PrivateEye": 24.9,
    "Qbert": 163.9,
    "RoadRunner": 11.5,
    "Seaquest": 68.4,
    "UpNDown": 533.4,
}


# Baseline comparison scores (from paper)
# These are the reported scores from SimPLe, CURL, DrQ, SPR
BASELINE_SCORES = {
    "SimPLe": {
        "Alien": 616.9,
        "Amidar": 74.3,
        "Assault": 527.2,
        "Asterix": 1128.3,
        "BankHeist": 34.2,
        "BattleZone": 4031.2,
        "Boxing": 7.8,
        "Breakout": 16.4,
        "ChopperCommand": 979.4,
        "CrazyClimber": 62583.6,
        "DemonAttack": 208.1,
        "Freeway": 16.7,
        "Frostbite": 236.9,
        "Gopher": 596.8,
        "Hero": 2656.6,
        "Jamesbond": 100.5,
        "Kangaroo": 51.2,
        "Krull": 2204.8,
        "KungFuMaster": 14862.5,
        "MsPacman": 1480.0,
        "Pong": 12.8,
        "PrivateEye": 35.0,
        "Qbert": 1288.8,
        "RoadRunner": 5640.6,
        "Seaquest": 683.3,
        "UpNDown": 3350.3,
    },
    "CURL": {
        "Alien": 711.0,
        "Amidar": 113.7,
        "Assault": 500.9,
        "Asterix": 567.2,
        "BankHeist": 65.3,
        "BattleZone": 8997.8,
        "Boxing": 0.9,
        "Breakout": 2.6,
        "ChopperCommand": 783.5,
        "CrazyClimber": 9154.4,
        "DemonAttack": 646.5,
        "Freeway": 28.3,
        "Frostbite": 1226.5,
        "Gopher": 400.9,
        "Hero": 4987.7,
        "Jamesbond": 331.0,
        "Kangaroo": 740.2,
        "Krull": 3049.2,
        "KungFuMaster": 8155.6,
        "MsPacman": 1064.0,
        "Pong": -18.5,
        "PrivateEye": 81.9,
        "Qbert": 727.0,
        "RoadRunner": 5006.1,
        "Seaquest": 315.2,
        "UpNDown": 2646.4,
    },
    "DrQ": {
        "Alien": 865.2,
        "Amidar": 137.8,
        "Assault": 579.6,
        "Asterix": 763.6,
        "BankHeist": 232.9,
        "BattleZone": 10165.3,
        "Boxing": 9.0,
        "Breakout": 19.8,
        "ChopperCommand": 844.6,
        "CrazyClimber": 21539.0,
        "DemonAttack": 1321.5,
        "Freeway": 20.3,
        "Frostbite": 1014.2,
        "Gopher": 621.6,
        "Hero": 4167.9,
        "Jamesbond": 349.1,
        "Kangaroo": 1088.4,
        "Krull": 4402.1,
        "KungFuMaster": 11467.4,
        "MsPacman": 1218.1,
        "Pong": -9.1,
        "PrivateEye": 3.5,
        "Qbert": 1810.7,
        "RoadRunner": 11211.4,
        "Seaquest": 352.3,
        "UpNDown": 4324.5,
    },
    "SPR": {
        "Alien": 841.9,
        "Amidar": 179.7,
        "Assault": 565.6,
        "Asterix": 962.5,
        "BankHeist": 345.4,
        "BattleZone": 14834.1,
        "Boxing": 35.7,
        "Breakout": 19.6,
        "ChopperCommand": 946.3,
        "CrazyClimber": 36700.5,
        "DemonAttack": 517.6,
        "Freeway": 19.3,
        "Frostbite": 1170.7,
        "Gopher": 660.6,
        "Hero": 5858.6,
        "Jamesbond": 366.5,
        "Kangaroo": 3617.4,
        "Krull": 3681.6,
        "KungFuMaster": 14783.2,
        "MsPacman": 1318.4,
        "Pong": -5.4,
        "PrivateEye": 86.0,
        "Qbert": 866.3,
        "RoadRunner": 12213.1,
        "Seaquest": 558.1,
        "UpNDown": 10859.2,
    },
}


def compute_human_normalized_score(score: float, game: str) -> float:
    """Compute human-normalized score for a game.

    Formula: (score - random) / (human - random)
    """
    human = HUMAN_SCORES.get(game, 1.0)
    random = RANDOM_SCORES.get(game, 0.0)

    if human == random:
        return 0.0

    return (score - random) / (human - random)


def run_single_game(
    game: str,
    config: IRISConfig = None,
    device: str = "cuda",
    num_seeds: int = 5,
) -> Dict:
    """Run IRIS on a single game with multiple seeds.

    Args:
        game: Game name (e.g., "ALE/Pong-v5")
        config: IRIS config
        device: Device to run on
        num_seeds: Number of random seeds to run

    Returns:
        Dictionary with results
    """
    print(f"\n{'=' * 60}")
    print(f"Training on: {game}")
    print(f"{'=' * 60}")

    results = []

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed + 1}/{num_seeds} ---")

        try:
            trainer = IRISTrainer(
                game=game,
                device=device,
                config=config,
                seed=seed,
            )

            # Train for full 100k steps
            trainer.train(total_epochs=config.total_epochs, eval_interval=100)

            # Evaluate
            eval_metrics = trainer.evaluate(num_episodes=100)
            results.append(eval_metrics["eval_mean_return"])

        except Exception as e:
            print(f"Error on seed {seed}: {e}")
            results.append(0.0)

    # Compute statistics
    mean_return = np.mean(results)
    std_return = np.std(results)

    # Get game name without ALE/ prefix
    game_name = game.replace("ALE/", "").replace("-v5", "")

    # Compute human-normalized score
    hns = compute_human_normalized_score(mean_return, game_name)

    result = {
        "game": game,
        "game_name": game_name,
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "human_normalized_score": float(hns),
        "individual_runs": results,
    }

    print(f"\nResults for {game_name}:")
    print(f"  Mean return: {mean_return:.2f} +/- {std_return:.2f}")
    print(f"  Human-normalized: {hns:.3f}")

    return result


def run_atari_100k(
    games: List[str] = None,
    config: IRISConfig = None,
    device: str = "cuda",
    output_file: str = "results/iris_atari100k.json",
    num_seeds: int = 5,
):
    """Run IRIS on all Atari 100k games.

    Args:
        games: List of games to run (default: all 26)
        config: IRIS config
        device: Device to run on
        output_file: Output JSON file for results
        num_seeds: Number of seeds per game
    """
    if games is None:
        games = ATARI_100K_GAMES

    if config is None:
        config = IRISConfig()

    results = []
    num_superhuman = 0

    for game in games:
        result = run_single_game(game, config, device, num_seeds)
        results.append(result)

        # Count superhuman games
        if result["human_normalized_score"] > 1.0:
            num_superhuman += 1

    # Compute aggregate metrics
    hns_scores = [r["human_normalized_score"] for r in results]

    # Compute baseline comparisons
    baseline_hns = {}
    for baseline_name, baseline_returns in BASELINE_SCORES.items():
        baseline_scores = []
        for r in results:
            game_name = r["game_name"]
            if game_name in baseline_returns:
                hns = compute_human_normalized_score(
                    baseline_returns[game_name], game_name
                )
                baseline_scores.append(hns)
            else:
                baseline_scores.append(0.0)

        baseline_hns[baseline_name] = {
            "mean": float(np.mean(baseline_scores)),
            "median": float(np.median(baseline_scores)),
            "iqm": float(np.mean(np.percentile(baseline_scores, [25, 75]))),
        }

    aggregate = {
        "num_games": len(results),
        "num_superhuman": num_superhuman,
        "mean_hns": float(np.mean(hns_scores)),
        "median_hns": float(np.median(hns_scores)),
        "iqm_hns": float(np.mean(np.percentile(hns_scores, [25, 75]))),
        "baseline_comparison": baseline_hns,
    }

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "results": results,
                "aggregate": aggregate,
            },
            f,
            indent=2,
        )

    # Print summary
    print(f"\n{'=' * 60}")
    print("ATARI 100K BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Games tested: {aggregate['num_games']}")
    print(f"Superhuman games: {aggregate['num_superhuman']}")
    print(f"Mean HNS: {aggregate['mean_hns']:.3f}")
    print(f"Median HNS: {aggregate['median_hns']:.3f}")
    print(f"IQM HNS: {aggregate['iqm_hns']:.3f}")
    print("\nComparison to baselines:")
    for name, stats in baseline_hns.items():
        print(
            f"  {name}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, iqm={stats['iqm']:.3f}"
        )
    print(f"\nResults saved to: {output_file}")

    return results, aggregate


def print_results_table(results: List[Dict]):
    """Print a nice table of results."""
    print(f"\n{'Game':<20} {'Mean Return':>12} {'Std':>8} {'HNS':>8}")
    print("-" * 50)

    for r in results:
        print(
            f"{r['game_name']:<20} {r['mean_return']:>12.1f} {r['std_return']:>8.1f} {r['human_normalized_score']:>8.3f}"
        )


def main():
    """Run full Atari 100k benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Run IRIS Atari 100k benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/iris_atari100k.json",
        help="Output file",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5, help="Number of random seeds per game"
    )
    parser.add_argument(
        "--games", type=str, nargs="+", default=None, help="Specific games to run"
    )

    args = parser.parse_args()

    # Create config optimized for Atari 100k
    config = IRISConfig()
    config.atari_100k = True
    config.max_env_steps = 100000

    # Run benchmark
    results, aggregate = run_atari_100k(
        games=args.games,
        config=config,
        device=args.device,
        output_file=args.output,
        num_seeds=args.num_seeds,
    )

    # Print results table
    print_results_table(results)


if __name__ == "__main__":
    main()
