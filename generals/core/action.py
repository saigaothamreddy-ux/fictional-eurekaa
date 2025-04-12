from typing import TypeAlias

import numpy as np

from generals.core.config import DIRECTIONS
from generals.core.observation import Observation

Action: TypeAlias = np.ndarray


def compute_valid_move_mask(observation: Observation) -> np.ndarray:
    """
    Return a mask of the valid moves for a given observation.

    A valid move originates from a cell the agent owns, has at least 2 armies on
    and does not attempt to enter a mountain nor exit the grid.

    A move is distinct from an action. A move only has 3 dimensions: (row, col, direction).
    Whereas an action also includes to_pass & to_split.

    Returns:
        np.ndarray: an NxNx4 array, where each channel is a boolean mask
        of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.

        I.e. valid_action_mask[i, j, k] is 1 if action k is valid in cell (i, j).
    """
    height, width = observation.owned_cells.shape

    ownership_channel = observation.owned_cells
    more_than_1_army = (observation.armies > 1) * ownership_channel
    owned_cells_indices = np.argwhere(more_than_1_army)
    valid_action_mask = np.zeros((height, width, 4), dtype=bool)

    if np.sum(ownership_channel) == 0:
        return valid_action_mask

    # check if destination is road
    passable_cells = 1 - observation.mountains

    for channel_index, direction in enumerate(DIRECTIONS):
        destinations = owned_cells_indices + direction.value

        # check if destination is in grid bounds
        in_first_boundary = np.all(destinations >= 0, axis=1)
        in_height_boundary = destinations[:, 0] < height
        in_width_boundary = destinations[:, 1] < width
        destinations = destinations[in_first_boundary & in_height_boundary & in_width_boundary]

        # assert that every value is either 0 or 1 in passable cells
        passable_cell_indices = passable_cells[destinations[:, 0], destinations[:, 1]] == 1
        action_destinations = destinations[passable_cell_indices]

        # get valid action mask for a given direction
        valid_source_indices = action_destinations - direction.value
        valid_action_mask[valid_source_indices[:, 0], valid_source_indices[:, 1], channel_index] = 1.0

    return valid_action_mask
