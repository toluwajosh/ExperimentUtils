"""
Training logging utils
Author: Joshua Owoyemi
Date: 2020-09-7
"""

import logging
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

logger = logging.getLogger(f"base.{__name__}")


def setup_logging(logger: logging.Logger, log_level: str, log_file: str = "") -> None:
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
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S",
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
        _l = getattr(logging, lvl)
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


# TODO: remove in update
def tensorboard_image(
    writer: SummaryWriter,
    image: torch.tensor,
    target: torch.tensor,
    output: torch.tensor,
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


# TODO: allow arbitrary number of images by taking input as a dict({name:image})
def tensorboard_images(writer: SummaryWriter, images: Dict, global_step: int,) -> None:
    """Output image tensors in tensorboard visualization. 
    Useful for image based training

    Args:
        writer (SummaryWriter): tensorboard writer object
        images (Dict): dictionary of {name: image_tensor} pair
        global_step (int): global step number
    """
    for name, image in images.items():
        grid_image = make_grid(image[:2].clone().cpu().data, 3, normalize=True)
        writer.add_image(name, grid_image, global_step)


if __name__ == "__main__":
    pass
