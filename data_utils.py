"""
Data utils
"""
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List

from sklearn.utils import shuffle

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
