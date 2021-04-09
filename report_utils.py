"""
utils for generating reports
"""

import json
from typing import Dict, Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import math
import cv2


def plot_conf_matrix(cm: np.ndarray, save_path: Optional[str]) -> plt.Figure:
    """Confusion matrix

    Args:
        cm ([np.ndarray]): confusion matrix array
        save_path ([str], optional): path to save the figure. Defaults to None.

    Returns:
        [plt.Figure]: confusion matrix figure
    """
    fig = plt.figure()
    plt.matshow(cm)
    plt.title("Pixels Confusion Matrix")
    plt.colorbar()
    plt.ylabel("True Label")
    plt.xlabel("Predicated Label")
    x, y = cm.shape
    for x_val in range(x):
        for y_val in range(y):
            c = round(cm[x_val, y_val], 3)
            plt.text(x_val, y_val, c, va="center", ha="center")
    if save_path is not None:
        plt.savefig(save_path)
    return fig


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
    max_size: Optional[int] = 1440,
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


class ResultData(object):
    def __init__(self, method_name: str, save_path: str = "./results") -> None:
        self.__method_name = method_name
        self.__save_path = save_path

        # Default containers
        self.__metrics = {}
        # self.__categories = []
        self.__summaries = {}

    def add_result(self, metric: str, category: Dict) -> None:
        """Add a result category

        Args:
            metric (str): the name of the result metric
            category (Dict): a dictionary of category_name: score
        """
        cur_metric_results = self.__metrics.get(metric, [])
        cur_metric_results.append(category)
        self.__metrics.update({metric: cur_metric_results})

    def add_summary(self, metric: str, summary: Dict) -> None:
        """Add a summary and value

        Args:
            summary (Dict): a dictionary of summary: value
        """
        if metric in self.__metrics:
            cur_summary = self.__summaries.get(metric, {})
            cur_summary.update(summary)
            self.__summaries.update({metric: cur_summary})
        else:
            raise ValueError(f"No metric by {metric}")

    def to_dict(self):
        assert len(self.__summaries) == len(
            self.__metrics
        ), "Size of summaries and metrics must be equal"
        return {
            "method": self.__method_name,
            "metrics": [
                {"metric": metric, "results": category, "summary": summary}
                for (metric, category, summary) in zip(
                    self.__metrics.keys(),
                    self.__metrics.values(),
                    self.__summaries.values(),
                )
            ],
        }

    def to_json(self):
        json_file = f"{self.__save_path}/{self.__method_name}.json"
        with open(json_file, "w") as file:
            json.dump(self.to_dict(), file)

    def __str__(self):
        out_string = f"*** {self.__method_name} Results ***\n"
        for item in self.__metrics:
            out_string += f"{str(item)}\n"
        return out_string

    @property
    def method_name(self):
        print("getting method name")
        return self.__method_name

    @property
    def save_path(self):
        print("getting save path")
        return self.__save_path


if __name__ == "__main__":
    # example:
    result_data = ResultData("new_example_result_2", "examples/results")
    result_data.add_result("scorer", {"category": "category 1", "score": 0.01})
    result_data.add_result("scorer", {"category": "category 2", "score": 0.03})
    result_data.add_summary("scorer", {"average": 90})
    result_data.add_result("scorer2", {"category": "category 3", "score": 0.06})
    result_data.add_result("scorer2", {"category": "category 3", "score": 0.06})
    result_data.add_summary("scorer2", {"average": 80})
    result_data.to_json()
    # print(result_data.to_dict())
    # print(result_data)
