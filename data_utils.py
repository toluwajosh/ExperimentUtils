"""
Data utils
"""
import logging
import math
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
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
        images (Union[np.ndarray, torch.Tensor]): Batch of images [batch, channels, width, height]
        fname (Optional[str], optional): Optional save name for mosaic. Defaults to "images.jpg".
        max_size (Optional[int], optional): Maximum single image size. Defaults to 640.
        max_subplots (Optional[int], optional): Maximum number of subplots. Defaults to 16.

    Returns:
        [np.ndarray]: mosaic image
    """
    # tl = 3  # line thickness
    # tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, c, h, w = images.shape  # batch size, _, height, width

    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    if c == 3:
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    else:
        mosaic = np.full((int(ns * h), int(ns * w)), 255, dtype=np.uint8)
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        if c == 3:
            img = img.transpose(1, 2, 0)
            if scale_factor < 1:
                img = cv2.resize(img, (w, h))

            mosaic[block_y : block_y + h, block_x : block_x + w, :] = img
        else:
            if scale_factor < 1:
                img = cv2.resize(img, (w, h))
            mosaic[block_y : block_y + h, block_x : block_x + w] = img
    if fname is not None:
        # mosaic = cv2.resize(
        #     mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA
        # )
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


def grid_pairs(
    bounds: List[int], resolution: List[int], dtype: Optional[str] = "float"
) -> List:
    """get grid pairs of patches in a larger array

    Args:
        bounds (List[int]): min_x, max_x, min_y, max_y of the lager array
        resolution (List[int]): size of patch [w,h]
        dtype (Optional[str], optional): output data type ["float", "int"]. Defaults to "float".

    Returns:
        [List]: output grid pairs
    """
    min_x, max_x, min_y, max_y = bounds
    if dtype == "float":
        out_type = float
    elif dtype == "int":
        out_type = int
    else:
        raise NotImplementedError
    # transform to origin
    nx = np.round((max_x - min_x) / resolution[0]).astype(np.int) + 1
    ny = np.round((max_y - min_y) / resolution[1]).astype(np.int) + 1
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    xv, yv = np.meshgrid(x, y)
    grid_points = list(zip(xv.flatten(), yv.flatten()))
    skips = int((max_x - min_x) / resolution[0]) + 2
    count = 0
    pairs_coords = []
    for i, point in enumerate(grid_points[:-skips]):
        if point[0] == max_x:
            continue
        count += 1
        pairs_coords.append(
            list(
                map(
                    out_type,
                    [
                        point[0],
                        point[1],
                        grid_points[i + skips][0],
                        grid_points[i + skips][1],
                    ],
                )
            )
        )
    return pairs_coords


def get_patch_boundaries(
    image_size: List[int],
    patch_size: List[int] = [256, 256],
    grids_dim: List[int] = None,
) -> List:
    """generate a list of image boundaries within a given image size

    Args:
        image_size (List[int]): image size
        patch_size (List[int]): patch size

    Returns:
        List: grid pairs, representing boundaris of smaller patches
    """
    assert len(image_size) == 2, "Len of image_size != 2."

    if grids_dim is not None:
        nx, ny = grids_dim
        patch_size = [x // y for (x, y) in zip(image_size, grids_dim)]
    else:
        nx, ny = [x // y for (x, y) in zip(image_size, patch_size)]

    limitx, limity = patch_size[0] * nx, patch_size[1] * ny
    boundaries = grid_pairs([0, limitx, 0, limity], patch_size, dtype="int")

    # TODO: Treat uncovered area.
    # In an implementation using the patch size,
    # the whole image is possibly not covered,
    # we can add extra code to cover left out areas, which may result in overlap
    return boundaries


def get_patches_batch(
    image: torch.tensor, grids_dim: List[int] = [5, 5]
) -> torch.tensor:
    """Split a large tensor into patches and return as a batch tensor data

    Args:
        image (torch.tensor): large image tensor
        grids_dim (List[int], optional): grids dimension. Defaults to [5, 5].

    Returns:
        torch.tensor: output patch tensor batch
    """
    _, _, w, h = image.shape
    patch_bounds = get_patch_boundaries([w, h], grids_dim=grids_dim)
    patches_list = []
    for patch in patch_bounds:
        x1, y1, x2, y2 = patch
        patch_tensor = image[:, :, x1:x2, y1:y2]
        patches_list.append(patch_tensor)
        patch_tensor = None
    return torch.cat(patches_list)


def per_class_split(
    data: List, labels: List, label_names: List, split: float
) -> Tuple[List]:
    """Split a set of data and labels by the size of each category

    Args:
        data (List): List of data samples
        labels (List): List of data labels
        label_names (List): Names given to each category
        split (float): the split size

    Returns:
        Tuple: The splitted data
    """
    assert isinstance(split, float), "Split argument should be a float type"

    samples_dict = {cat: [] for cat in label_names}
    labels_dict = {cat: [] for cat in label_names}
    for sample, label in zip(data, labels):
        samples_dict[label].append(sample)
        labels_dict[label].append(label)
    a_data_split = []
    a_labels_split = []
    b_data_split = []
    b_labels_split = []
    for label in label_names:
        cur_split = int(split * len(samples_dict[label]))
        a_data_split += samples_dict[label][:cur_split]
        a_labels_split += labels_dict[label][:cur_split]
        b_data_split += samples_dict[label][cur_split:]
        b_labels_split += labels_dict[label][cur_split:]
    return (a_data_split, a_labels_split), (b_data_split, b_labels_split)


if __name__ == "__main__":
    # images = np.full([3, 3, 128, 256], 0, np.uint8)
    # mosaic = plot_images_mosaic(images)
    image_shape = [1, 3, 2880, 2160]
    image = torch.ones(image_shape)
    patches_tensor = get_patches_batch(image)
    print(patches_tensor.shape)

    pass
