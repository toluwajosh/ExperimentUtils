import torch
import logging
from pathlib import Path
from typing import Dict, Union

logger = logging.getLogger(f"base.{__name__}")


def load_checkpoint(
    model: torch.nn.Module, save_path: Path, exit_on_fail: bool = False,
) -> Dict:
    """Load model checkpoint from path,
        Also loads the model state dict

    Args:
        model (torch.nn.Module): torch model
        save_path (str): path to the model checkpoint
        exit_on_fail (bool): Flag to exit program if loading failes.
    """
    if save_path.is_file():
        save_path_name = str(save_path)
        checkpoint = torch.load(save_path_name)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded checkpoint: {save_path_name}")
    else:
        if exit_on_fail:
            raise ValueError("No checkpoint found at '{}'".format(save_path))
        logger.warn("=> no checkpoint found at '{}'".format(save_path))
        checkpoint = {"epoch": 1, "best": float("inf")}

    return checkpoint


def save_checkpoint(state: Dict, filename: Union[str, Path], is_best: bool) -> None:
    """Save checkpoint if a new best is achieved

    Args:
        state (dict): torch model dictionary
        filename (Union[str, Path]): filename to save
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


if __name__ == "__main__":
    pass
