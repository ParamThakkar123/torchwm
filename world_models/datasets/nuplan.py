import os
import json
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np

# Note: This implementation assumes the NuPlan dataset is available locally.
# NuPlan dataset can be downloaded from https://www.nuplan.org/nuplan
# and the nuplan-devkit should be installed.

# For full integration, we need:
# - nuplan-devkit for data loading
# - Path to the dataset root
# - Scenario filtering (e.g., by location, vehicle type)
# - Data preprocessing (map rendering, agent trajectories, ego history)


class NuPlanDataset(Dataset):
    """
    PyTorch Dataset for NuPlan autonomous driving dataset.

    Loads scenarios and provides planning tasks for world model training.
    Each sample contains:
    - Map data (rasterized or vector)
    - Agent trajectories (past and future)
    - Ego vehicle state
    - Planning task (e.g., predict future trajectory)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        scenario_filter: Optional[Dict] = None,
        map_version: str = "nuplan-maps-v1.0",
        sensor_config: Optional[Dict] = None,
        planning_horizon: int = 80,  # 8 seconds at 10Hz
        past_horizon: int = 20,  # 2 seconds
        map_extent: Tuple[float, float] = (100.0, 100.0),  # meters
        map_resolution: float = 0.1,  # meters per pixel
    ):
        """
        Args:
            root_dir: Path to NuPlan dataset root
            split: Dataset split ('train', 'val', 'test')
            scenario_filter: Dict of filters for scenarios (location, vehicle_type, etc.)
            map_version: Version of maps to use
            sensor_config: Configuration for sensor data loading
            planning_horizon: Number of future timesteps to predict (10Hz)
            past_horizon: Number of past timesteps to include
            map_extent: Size of map crop around ego (width, height in meters)
            map_resolution: Resolution of rasterized map
        """
        self.root_dir = root_dir
        self.split = split
        self.scenario_filter = scenario_filter or {}
        self.map_version = map_version
        self.sensor_config = sensor_config
        self.planning_horizon = planning_horizon
        self.past_horizon = past_horizon
        self.map_extent = map_extent
        self.map_resolution = map_resolution

        # TODO: Initialize nuplan-devkit data access
        # from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
        # from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

        # Load scenario database
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self) -> List[Dict]:
        """Load list of scenarios for the dataset split."""
        # TODO: Implement scenario loading using nuplan-devkit
        # This should query the database for scenarios matching filters
        # Return list of scenario metadata

        # Placeholder implementation
        scenarios_file = os.path.join(self.root_dir, f"{self.split}_scenarios.json")
        if os.path.exists(scenarios_file):
            with open(scenarios_file, "r") as f:
                return json.load(f)
        else:
            # Return empty list if file doesn't exist
            return []

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Dict:
        """Load a single scenario sample."""
        scenario_meta = self.scenarios[idx]

        # TODO: Load scenario data using nuplan-devkit
        # scenario = self._load_scenario(scenario_meta)

        # Placeholder data
        sample = {
            "scenario_id": scenario_meta.get("scenario_id", f"scenario_{idx}"),
            "map": self._load_map(scenario_meta),
            "ego_trajectory": self._load_ego_trajectory(scenario_meta),
            "agent_trajectories": self._load_agent_trajectories(scenario_meta),
            "planning_target": self._create_planning_target(scenario_meta),
        }

        return sample

    def _load_map(self, scenario_meta: Dict) -> torch.Tensor:
        """Load and rasterize map data around ego position."""
        # TODO: Implement map loading and rasterization
        # - Load map layers (lanes, crosswalks, etc.)
        # - Crop around current ego position
        # - Rasterize to tensor

        # Placeholder: Return random map tensor
        map_channels = 3  # drivable, lanes, crosswalks
        map_size = (
            int(self.map_extent[0] / self.map_resolution),
            int(self.map_extent[1] / self.map_resolution),
        )
        return torch.randn(map_channels, map_size[0], map_size[1])

    def _load_ego_trajectory(self, scenario_meta: Dict) -> Dict[str, torch.Tensor]:
        """Load ego vehicle past and future trajectory."""
        # TODO: Load ego pose history and future ground truth

        # Placeholder
        past_trajectory = torch.randn(
            self.past_horizon, 6
        )  # x, y, yaw, vx, vy, yaw_rate
        future_trajectory = torch.randn(
            self.planning_horizon, 2
        )  # x, y relative to current

        return {
            "past": past_trajectory,
            "future": future_trajectory,
        }

    def _load_agent_trajectories(
        self, scenario_meta: Dict
    ) -> List[Dict[str, torch.Tensor]]:
        """Load trajectories of other agents in the scene."""
        # TODO: Load agent states and trajectories

        # Placeholder: Return list of agent trajectories
        num_agents = np.random.randint(0, 20)
        agents = []
        for _ in range(num_agents):
            agent = {
                "past_trajectory": torch.randn(self.past_horizon, 6),
                "future_trajectory": torch.randn(self.planning_horizon, 6),
                "agent_type": torch.randint(0, 5, (1,)),  # vehicle, pedestrian, etc.
            }
            agents.append(agent)

        return agents

    def _create_planning_target(self, scenario_meta: Dict) -> torch.Tensor:
        """Create planning target for the scenario."""
        # TODO: Extract planning task (e.g., future trajectory prediction)

        # Placeholder: Return future ego trajectory as target
        return torch.randn(self.planning_horizon, 2)


def make_nuplan_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
):
    """Create DataLoader for NuPlan dataset."""
    dataset = NuPlanDataset(root_dir=root_dir, split=split, **dataset_kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_nuplan_batch,
    )

    return dataset, dataloader


def collate_nuplan_batch(batch: List[Dict]) -> Dict:
    """Collate function for NuPlan batches."""
    # TODO: Implement proper batch collation
    # Handle variable number of agents, pad sequences, etc.

    collated = {}
    for key in batch[0].keys():
        if key == "agent_trajectories":
            # Special handling for variable-length agent lists
            collated[key] = [sample[key] for sample in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], dict):
            collated[key] = {
                subkey: torch.stack([sample[key][subkey] for sample in batch])
                for subkey in batch[0][key].keys()
            }
        else:
            collated[key] = [sample[key] for sample in batch]

    return collated
