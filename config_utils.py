from copy import deepcopy
from pathlib import Path
from typing import Dict, Type, TypeVar, Union

import yaml

# Allow to type hint to return the class itself
# Note, you can skip that if you're from the future (see PEP 563), or
# using: "from __future__ import annotations"
T = TypeVar("T", bound="ExpConfig")


class ExpConfig:
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
        return cls(**d)

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
