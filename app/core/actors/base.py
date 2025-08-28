"""
Base actor interface for power management agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Actor(ABC):
    """
    Abstract base class for power management actors (agents).

    An actor receives observations about the environment state and
    returns actions to control the power plant portfolio.
    """

    @abstractmethod
    def get_action(
        self, observation: Dict[str, Any], timestamp: Any = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the action to take given the current observation.

        Args:
            observation: Current environment observation
            timestamp: Current simulation timestamp (optional)

        Returns:
            Action dictionary in format:
            {
                "portfolio_name": {
                    "plant_name": {
                        "net_power_target_mw": float,
                        "battery_power_target_mw": float,  # optional
                        "pv_curtailment_factor": float,    # optional (0.0-1.0)
                    }
                }
            }
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
