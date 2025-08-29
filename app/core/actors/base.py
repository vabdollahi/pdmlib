"""
Base actor interface for power management agents.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Actor(ABC):
    """
    Abstract base class for power management actors (agents).

    An actor receives observations about the environment state and
    returns actions to control the power plant portfolio.

    All actors must be compatible with the gymnasium interface:
    - Observations: structured dictionaries
    - Actions: numpy arrays in [-1, 1] range
    """

    @abstractmethod
    def get_action(self, observation: dict, timestamp: Any = None) -> np.ndarray:
        """
        Get the action to take given the current observation.

        Args:
            observation: Current environment observation as structured dict
            timestamp: Current simulation timestamp (optional)

        Returns:
            Action as numpy array with values in [-1, 1] range.
            Each element corresponds to a plant's net power target:
            - -1.0: minimum net power (full consumption/curtailment)
            - +1.0: maximum net power (full generation + discharge)
            -  0.0: middle of the range
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the actor state (if any).

        Called at the beginning of each episode.
        """
        pass

    def step(self) -> None:
        """
        Update actor state after taking an action.

        Called after each environment step.
        """
        pass
