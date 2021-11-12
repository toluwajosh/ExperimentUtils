"""
Training logging utils
Author: Joshua Owoyemi
Date: 2020-09-7
"""

import datetime
import time
import logging
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

logger = logging.getLogger(f"base.{__name__}")


class ControlledLogging:
    def __init__(
        self, name: str, log_level: str = "DEBUG", log_file: str = "", rank: int = 0
    ) -> None:
        self.name = name
        self.rank = rank
        self.logger = logging.getLogger(self.name)
        self.setup(self.logger, log_level, log_file=log_file)

    @staticmethod
    def setup(logger: logging.Logger, log_level: str, log_file: str = "") -> None:
        """Setup logging for console output

        Args:
            logger (logging.Logger): the logger object
            log_level (str): the minimum level of logging
            log_file (optional, str): file to write logging output. Defaults to ""
        """

        # Setup console handler
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        logging_format = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d, %H:%M:%S",
        )
        ch.setFormatter(logging_format)
        logger.addHandler(ch)
        if log_file:
            Path(log_file).parent.mkdir(exist_ok=True, parents=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(logging_format)
            logger.addHandler(fh)

        # Add colors
        _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
        for color, lvl in _levels:
            _l = getattr(logging, lvl)  # type: ignore
            logging.addLevelName(
                _l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l))
            )

    def debug(self, message: str) -> None:
        if self.rank == 0:
            self.logger.debug(message)

    def info(self, message: str) -> None:
        if self.rank == 0:
            self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)


# TODO: Deprecate
def setup_logging(logger: logging.Logger, log_level: str, log_file: str = "") -> None:
    """Setup logging for console output

    Args:
        logger (logging.Logger): the logger object
        log_level (str): the minimum level of logging
        log_file (optional, str): file to write logging output. Defaults to ""
    """

    logger.warning(
        "This function will be deprecated. Use ControlLogging.setup instead!"
    )

    # Setup console handler
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logging_format = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d, %H:%M:%S",
    )
    ch.setFormatter(logging_format)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(logging_format)
        logger.addHandler(fh)

    # Add colors
    _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
    for color, lvl in _levels:
        _l = getattr(logging, lvl)  # type: ignore
        logging.addLevelName(
            _l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l))
        )


def setup_dirs(root_dir: Path, subdirectories: List[str]) -> None:
    """Create subdirectories from a list

    Args:
        root_dir (Path): main directory
        subdirectories (list): list of subdirectories
    """
    for subdirectory in subdirectories:
        new_dir = Path(root_dir) / Path(subdirectory)
        new_dir.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Get human readable timestamp
    Returns:
        str: timestamp of the form %Y_%m_%d_%H_%M_%S
    """
    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)
    return f"{value:%Y_%m_%d_%H_%M_%S}"


# TODO: remove in update
def tensorboard_image(
    writer: SummaryWriter,
    image: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
    global_step: int,
) -> None:
    """Output image tensors in tensorboard visualization. 
    Useful for image based training

    Args:
        writer (SummaryWriter): tensorboard writer object
        image (torch.tensor): image tensor
        target (torch.tensor): target image tensor
        output (torch.tensor): prediction image tensor
        global_step (int): global step number
    """
    # input image
    grid_image = make_grid(image[:2].clone().cpu().data, 3, normalize=True)
    writer.add_image("Input image", grid_image, global_step)

    # Model output
    grid_image = make_grid(output[:2].clone().cpu().data, 3, normalize=True)
    writer.add_image("Prediction", grid_image, global_step)

    # target image
    grid_image = make_grid(target[:2].clone().cpu().data, 3, normalize=True)
    writer.add_image("Target", grid_image, global_step)


def tensorboard_images(
    writer: SummaryWriter, images: Dict, global_step: int, normalize_flags=None
) -> None:
    """Output image tensors in tensorboard visualization. 
    Useful for image based training

    Args:
        writer (SummaryWriter): tensorboard writer object
        images (Dict): dictionary of {name: image_tensor} pair
        global_step (int): global step number
    """
    for i, (name, image) in enumerate(images.items()):
        if normalize_flags is None:
            normalize = True
        else:
            normalize = normalize_flags[i]
        grid_image = make_grid(image[:2].clone().cpu().data, 3, normalize=normalize)
        writer.add_image(name, grid_image, global_step)


if __name__ == "__main__":
    pass
