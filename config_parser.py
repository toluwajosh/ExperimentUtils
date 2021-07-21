from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Dict, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, validator

# Allow to type hint to return the class itself
# Note, you can skip that if you're from the future (see PEP 563), or
# using: "from __future__ import annotations"
T = TypeVar("T", bound="ExpConfig")


class BaseConfigModel(BaseModel):
    """Base class for experiment configurations. To hold variables for experiments
    """

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
        return cls(**d)  # type: ignore

    def to_yaml(self, path: Union[Path, str]) -> None:
        """Save a configuration to disk as yaml

        Args:
            path (Union[Path, str]): path to save
        """

        with Path(path).open("w") as f:
            yaml.safe_dump(self.to_dict(), f, indent=4)

    def to_dict(self) -> Dict:
        """Return a `dict` of the parameters with classes transformed into string."""

        # Need a copy not to overwrite itself
        d = deepcopy(vars(self))
        return d

    def __str__(self) -> str:
        output = "Experiment Configuration:\n"
        for k, v in vars(self).items():
            output += f"\t{k}: {v}\n"
        return output


class CLOptions:
    """
    Get command line options, based on the argparse library.
    A default option is the configuration file path (.config_path)
    Get the config by simply:
    config_path = CLOptions().parse().config_path
    """

    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument("config_path")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse(self) -> dict:
        return self.parser.parse_args()
