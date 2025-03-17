import numpy as np

from generals.core.config import DIRECTIONS, Direction

from .observation import Observation


class Action(np.ndarray):
    """
    Action objects walk & talk like typical numpy-arrays, but have a more descriptive and narrow interface.
    """

    def __new__(cls, to_pass: bool, row: int = 0, col: int = 0, direction: int | Direction = 0, to_split: bool = False):
        """
        Args:
            cls: This argument is automatically provided by Python and is the Action class.
            to_pass: Indicates whether the agent should pass/skip this turn i.e. do nothing. If to_pass is True,
                the other arguments, like row & col, are effectively ignored.
            row: The row the agent should move from. In the closed-interval: [0, (grid_height - 1)].
            col: The column the agent should move from. In the closed-interval: [0, (grid_width - 1)].
            direction: The direction the agent should move from the tile (row, col). Can either pass an enum-member
                of Directions or the integer representation of the direction, which is the relevant index into the
                config.DIRECTIONS array.
            to_split: Indicates whether the army in (row, col) should be split, then moved in direction.
        """
        if isinstance(direction, Direction):
            direction = DIRECTIONS.index(direction)

        # Note: np.array.view casts the np.array object to type cls, i.e. Action, without modifying
        # any of the arrays internal representation.
        action_array = np.array([to_pass, row, col, direction, to_split], dtype=np.int8).view(cls)
        return action_array

    def is_pass(self) -> bool:
        return self[0] == 1

    def is_split(self) -> bool:
        return self[4] == 1

    def __str__(self) -> str:
        if self.is_pass():
            return "Action(pass)"

        direction_str = DIRECTIONS[self[3]].name.lower()
        row, col = self[1], self[2]
        if self.is_split():
            return f"Action(split-move {direction_str} from ({row}, {col}))"
        return f"Action(move {direction_str} from ({row}, {col}))"

    def __repr__(self) -> str:
        return str(self)


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
        
        # Filter destinations that are passable
        passable_destinations = destinations[passable_cell_indices]
        
        if len(passable_destinations) > 0:
            # Get source cells corresponding to passable destinations
            # source_cells = passable_destinations - direction.value
            
            # # Get armies at source and destination
            # source_armies = observation.armies[source_cells[:, 0], source_cells[:, 1]]
            # dest_armies = observation.armies[passable_destinations[:, 0], passable_destinations[:, 1]]
            
            # Check if destination is a city or general with enough armies to block
            # is_city = observation.cities[passable_destinations[:, 0], passable_destinations[:, 1]] == 1
            # is_general = observation.generals[passable_destinations[:, 0], passable_destinations[:, 1]] == 1
            # is_opponent = observation.opponent_cells[passable_destinations[:, 0], passable_destinations[:, 1]] == 1
            # is_neutral = observation.neutral_cells[passable_destinations[:, 0], passable_destinations[:, 1]] == 1
            # has_enough_army = dest_armies - 1 >= source_armies
            
            # A destination is valid if it's not:
            # (a city AND neutral AND has enough army) OR
            # (opponent owned AND (city OR general) AND has enough army)
            # cant_pass_expensive_city = is_city & is_neutral & has_enough_army
            # cant_pass_expensive_enemy = is_opponent & is_general & has_enough_army
            # valid_destination_mask = ~(cant_pass_expensive_enemy)
            # valid_destinations = passable_destinations[valid_destination_mask]
            
            # get valid action mask for a given direction
            if len(passable_destinations) > 0:
                valid_source_indices = passable_destinations - direction.value
                valid_action_mask[valid_source_indices[:, 0], valid_source_indices[:, 1], channel_index] = 1.0

    return valid_action_mask
