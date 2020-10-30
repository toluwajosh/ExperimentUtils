"""
Option utils
Author: Joshua Owoyemi
Date: 2020-09-7
"""
import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type, TypeVar, Union, Any

import yaml

# Allow to type hint to return the class itself
# Note, you can skip that if you're from the future (see PEP 563), or
# using: "from __future__ import annotations"
T = TypeVar("T", bound="TrainerConfig")

logger = logging.getLogger(f"base.{__name__}")


@dataclass
class TrainerConfig:
    """Create a dataclass for options.

    Default Members:
        experiment_name: str
        epochs: int
        learning_rate: float
        device: str
        restart: bool
        lr_patience: int
        weight_decay: float
        scheduler_factor: float
    
    Member Functions
        from_yaml
        to_yaml
        _to_dict
    
    It can be used with the TrainOptions class to convert command line arguments to dataclass
    """

    epochs: int
    restart: bool
    device: str
    experiment_name: str
    learning_rate: float
    lr_patience: int
    weight_decay: float
    scheduler_factor: float

    @classmethod
    def from_yaml(cls: Type[T], path: Union[Path, str]) -> T:
        """Read yaml file

        Args:
            cls (Type[T]): experiment config class
            path (Union[Path, str]): path to yaml file

        Returns:
            T: config class
        """
        with Path(path).open("r") as f:
            d = yaml.safe_load(f)
        return cls(**d)

    def to_yaml(self, path: Union[Path, str]) -> None:
        """Save a configuration to disk as yaml

        Args:
            path (Union[Path, str]): path to save
        """

        with Path(path).open("w") as f:
            yaml.safe_dump(self._to_dict(), f, indent=4)

    def _to_dict(self) -> Dict:
        """Return a `dict` of the parameters with classes transformed into string."""

        # Need a copy not to overwrite itself
        d = deepcopy(vars(self))
        return d


class TrainOptions(object):
    def __init__(self, description: str = "") -> None:
        super(TrainOptions, self).__init__()
        if not description:
            description = "Training Options"

        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # type: ignore
        )

        self.parser.add_argument(
            "-e", "--epochs", type=int, default=1000, help="Number of epochs."
        )

        self.parser.add_argument(
            "-n",
            "--experiment_name",
            type=str,
            default="unnamed",
            help="Name of the experiment",
        )

        self.parser.add_argument(
            "--config_path",
            type=str,
            default="",
            help="Path to configuration file (.yaml)",
        )

        self.parser.add_argument(
            "-r",
            "--learning_rate",
            type=float,
            default=0.001,
            help="Learning rate of the optimizer",
        )

        self.parser.add_argument(
            "-p",
            "--lr_patience",
            type=int,
            default=5,
            help="Learning patience rate of the optimizer scheduler",
        )

        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=5e-4,
            help="Weight decay of the optimizer parameters",
        )

        self.parser.add_argument(
            "--scheduler_factor",
            type=float,
            default=0.89,
            help="Learning rate decay factor set be the scheduler",
        )

        self.parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="0",
            help="Specify a device number in case of multiple GPU training.",
        )

        self.parser.add_argument(
            "--log_level",
            type=str,
            default="debug",
            choices=["debug", "info", "warning", "error"],
            help="Logging level.",
        )

        self.parser.add_argument(
            "--restart", action="store_true", help="Flag for restarting training."
        )

        self.config = TrainerConfig

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse(self) -> dict:
        return vars(self.parser.parse_args())


if __name__ == "__main__":
    pass
