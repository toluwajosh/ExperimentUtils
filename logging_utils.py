"""
Training logging utils
Author: Joshua Owoyemi
Date: 2020-09-7
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

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


def save_checkpoint(state: Dict, filename: Union[str, Path], is_best: bool) -> None:
    """Save checkpoint if a new best is achieved

    Args:
        state (dict): torch model dictionary
        filename (str): filename to save
        is_best (bool): flag for best model yet
    """
    if is_best:
        logger.info("=> Saving new checkpoint")
        torch.save(state, filename)

        # save a separate file for the best model
        torch.save(state, str(filename) + ".best")
    else:
        torch.save(state, filename)
        logger.info("=> Validation Accuracy did not improve")


def load_checkpoint(
    model: torch.nn.Module,
    save_path: Path,
    load_best: bool = False,
    exit_on_fail: bool = False,
) -> Dict:
    """Load model checkpoint from path,
        Also loads the model state dict

    Args:
        model (torch.nn.Module): torch model
        save_path (str): path to the model checkpoint
    """
    if save_path.is_file():
        # TODO: consider a less error prone method of loading the best model.
        if load_best:
            save_path_name = str(save_path) + ".best"
            checkpoint = torch.load(save_path_name)
        else:
            save_path_name = str(save_path)
            checkpoint = torch.load(save_path_name)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded checkpoint: {save_path_name}")
    else:
        if exit_on_fail:
            raise ValueError("No checkpoint found at '{}'".format(save_path))
        logger.warn("=> no checkpoint found at '{}'".format(save_path))
        checkpoint = {"epoch": 0, "best": float("inf")}

    return checkpoint


def setup_dirs(root_dir: Path, subdirectories: List[str]) -> None:
    """Create subdirectories from a list

    Args:
        root_dir (Path): main directory
        subdirectories (list): list of subdirectories
    """
    for subdirectory in subdirectories:
        new_dir = Path(root_dir) / Path(subdirectory)
        new_dir.mkdir(parents=True, exist_ok=True)


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

    # target image
    grid_image = make_grid(target[:2].clone().cpu().data, 3, normalize=True)
    writer.add_image("Target", grid_image, global_step)

    # Model output
    grid_image = make_grid(output[:2].clone().cpu().data, 3, normalize=True)
    writer.add_image("Prediction", grid_image, global_step)


if __name__ == "__main__":
    pass
