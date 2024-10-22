import numpy as np
from scipy.ndimage import maximum_filter  # type: ignore

from .config import MOUNTAIN, PASSABLE

valid_generals = ["A", "B"]  # Generals are represented by A and B


class Channels:
    """
    army - army size in each cell
    general - general mask (1 if general is in cell, 0 otherwise)
    mountain - mountain mask (1 if cell is mountain, 0 otherwise)
    city - city mask (1 if cell is city, 0 otherwise)
    passable - passable mask (1 if cell is passable, 0 otherwise)
    ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
    ownership_neutral - ownership mask for neutral cells that are
    passable (1 if cell is neutral, 0 otherwise)
    """

    def __init__(self, grid: np.ndarray, _agents: list[str]):
        self._army: np.ndarray = np.where(np.isin(grid, valid_generals), 1, 0).astype(int)
        self._general: np.ndarray = np.where(np.isin(grid, valid_generals), 1, 0).astype(bool)
        self._mountain: np.ndarray = np.where(grid == MOUNTAIN, 1, 0).astype(bool)
        self._city: np.ndarray = np.where(np.char.isdigit(grid), 1, 0).astype(bool)
        self._passable: np.ndarray = (grid != MOUNTAIN).astype(bool)

        self._ownership: dict[str, np.ndarray] = {
            "neutral": ((grid == PASSABLE) | (np.char.isdigit(grid))).astype(bool)
        }
        for i, agent in enumerate(_agents):
            self._ownership[agent] = np.where(grid == chr(ord("A") + i), 1, 0).astype(bool)

        # City costs are 40 + digit in the cell
        city_costs = np.where(np.char.isdigit(grid), grid, "0").astype(int)
        self.army += 40 * self.city + city_costs

    def get_visibility(self, agent_id: str) -> np.ndarray:
        channel = self._ownership[agent_id]
        return maximum_filter(channel, size=3)

    @property
    def ownership(self) -> dict[str, np.ndarray]:
        return self._ownership

    @ownership.setter
    def ownership(self, value):
        self._ownership = value

    @property
    def army(self) -> np.ndarray:
        return self._army

    @army.setter
    def army(self, value):
        self._army = value

    @property
    def general(self) -> np.ndarray:
        return self._general

    @general.setter
    def general(self, value):
        self._general = value

    @property
    def mountain(self) -> np.ndarray:
        return self._mountain

    @mountain.setter
    def mountain(self, value):
        self._mountain = value

    @property
    def city(self) -> np.ndarray:
        return self._city

    @city.setter
    def city(self, value):
        self._city = value

    @property
    def passable(self) -> np.ndarray:
        return self._passable

    @passable.setter
    def passable(self, value):
        self._passable = value

    @property
    def ownership_neutral(self) -> np.ndarray:
        return self._ownership["neutral"]

    @ownership_neutral.setter
    def ownership_neutral(self, value):
        self._ownership["neutral"] = value
