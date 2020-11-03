"""
Data utils
"""
import logging
import math
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

# import cv2s
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torch.utils import data
from torch.utils.data import DataLoader

logger = logging.getLogger(f"base.{__name__}")


def rearrange(original: List, indices: List) -> List:
    """rearrange a list according to given index

    Args:
        original (List): original list 
        indices (List): indices for arrangement

    Returns:
        List: rearranged list
    """
    res = deepcopy(original)
    for i in enumerate(indices):
        res[i[1]] = original[i[0]]
    return res


# TODO: #1 Conisder storing patterns in an os-specific-persistent temporary directory
def shuffle_on_pattern(data_list: List, name: str = "shuffle_pattern") -> List:
    """shuffle a datalist based on a particular pattern

    Args:
        data_list (List): List of data samples
        name (str): Given name of pattern. In case of multiple patterns

    Returns:
        List: shuffled list of data samples
    """
    # check if rearrange pattern already exist
    r_pattern_path = Path("rearrange_pattern.pkl")
    if not r_pattern_path.exists():
        # create pattern and save
        r_pattern = shuffle(range(len(data_list)))
        with open(r_pattern_path, "wb") as f:
            pickle.dump(r_pattern, f)
        logger.debug("Created new dataset shuffle pattern!")
    else:
        # load pattern
        with open(r_pattern_path, "rb") as f:
            r_pattern = pickle.load(f)
        logger.debug("Loaded previous dataset shuffle pattern!")
    return rearrange(data_list, r_pattern)


def plot_image_single(
    image: np.ndarray, size: Optional[int] = 15, title: Optional[str] = ""
) -> None:
    """Plot a single image using matplotlib, Without overwriting a privious.

    Args:
        image (np.ndarray): image to plot
        size (Optional[int]): size of the plot image
        title (Optional[str]): optional title given to the plot
    """
    f = plt.figure(figsize=(size, size))
    image_plot = f.add_subplot()
    if title:
        image_plot.set_title(title)
    plt.imshow(image)


def plot_images_mosaic(
    images: Union[np.ndarray, torch.Tensor],
    fname: Optional[str] = "images.jpg",
    max_size: Optional[int] = 640,
    max_subplots: Optional[int] = 16,
):
    """Plot a batch of images as a mosaic stitched together.

    Args:
        images (Union[np.ndarray, torch.Tensor]): Batch of images
        fname (Optional[str], optional): Optional save name for mosaic. Defaults to "images.jpg".
        max_size (Optional[int], optional): Maximum single image size. Defaults to 640.
        max_subplots (Optional[int], optional): Maximum number of subplots. Defaults to 16.

    Returns:
        [np.ndarray]: mosaic image
    """
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width

    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y : block_y + h, block_x : block_x + w, :] = img
    if fname is not None:
        mosaic = cv2.resize(
            mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    return mosaic


def prepare_dataset(
    dataloader: data.Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Prepare dataset for training

    Args:
        dataloader (data.Dataset)
        batch_size (int)
        shuffle (bool)
        num_workers (int)

    Returns:
        DataLoader
    """
    params = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
    }
    return DataLoader(dataloader, **params)


if __name__ == "__main__":
    images = np.full([3, 3, 128, 256], 0, np.uint8)
    mosaic = plot_images_mosaic(images)
