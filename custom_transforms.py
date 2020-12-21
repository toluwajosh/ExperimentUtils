"""
Custom transformations for images used in a Fully convolutional model
"""

import random
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class DummyTransformation(object):
    """Dummy transformation for arbitrary model input and target"""

    def __init__(*args, **kwargs):
        pass

    def __call__(self, sample: Dict, include_targets: bool) -> NotImplementedError:
        image = sample["image"]
        targets = sample["targets"]

        # Apply transformation on image
        pass

        return_dict = {"image": image}
        if include_targets:
            # apply same transformation on targets
            pass
        return_dict.update(targets)
        # return return_dict
        return NotImplementedError


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    @staticmethod
    def __call__(sample: Dict) -> Dict:
        """Convert ndarrays in sample to Tensors
            swap color axis for ndim = 3
            numpy image: H x W x C
            torch image: C X H X W

        Args:
            sample (Dict): Dictionary of ndarray samples

        Returns:
            Union[Dict, NotImplementedError]: returns a dictionary of converted tensors. 
                A NotImplementedError is raise if dimension of ndarry is not 2 or 3
        """

        image = sample["image"]
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        targets = sample["targets"]
        new_targets = {}
        for key, data in targets.items():
            data = np.array(data).astype(np.float32)
            if data.ndim == 3:
                data = data.transpose((2, 0, 1))
            data = torch.from_numpy(data).float()
            new_targets[key] = data
        return {"image": image, "targets": new_targets}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
        The input image is only normalized. Target are unchanged.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"]
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        targets = sample["targets"]
        # return_dict = {"image": image}
        targets = {key: value for key, value in targets.items()}
        # return_dict.update(targets)
        # return return_dict
        return {"image": image, "targets": targets}


class RandomCropRect(object):
    """
    Random croping to square size
    It takes great advantage in securing width-wise wide crop
    Expects a PIL Image
    """

    def __init__(self, crop_size):
        self.width = crop_size[0]
        self.height = crop_size[1]

    def __call__(self, sample):
        image = sample["image"]
        w, h = image.size
        x = random.randint(0, w - self.width)
        y = random.randint(0, h - self.height)
        image = image.crop((x, y, x + self.width, y + self.height))
        # return_dict = {"image": image}
        targets = sample["targets"]
        targets = {
            key: tensor.crop((x, y, x + self.width, y + self.height))
            for key, tensor in targets.items()
        }
        # return_dict.update(targets)
        return {"image": image, "targets": targets}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample["image"]
        if "label" in sample:
            mask = sample["label"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if "label" in sample:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if "label" in sample:
            return {"image": img, "label": mask, "name": sample["name"]}

        return {"image": img, "name": sample["name"]}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample["image"]
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        if "label" in sample:
            mask = sample["label"]
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            return {"image": img, "label": mask, "name": sample["name"]}

        return {"image": img, "name": sample["name"]}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample["image"]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return_dict = {"image": img, "name": sample["name"]}
        if "label" in sample:
            return_dict.update({"label": sample["label"]})
        if "lanes" in sample:
            return_dict.update({"lanes": sample["lanes"]})

        return return_dict


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        # random scale (short edge)
        short_size = random.randint(
            int(self.base_size * 0.5), int(self.base_size * 2.0)
        )
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        if "label" in sample:
            mask = sample["label"]
            mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            if "label" in sample:
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if "label" in sample:
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            return {"image": img, "label": mask, "name": sample["name"]}

        return {"image": img, "name": sample["name"]}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.0))
        y1 = int(round((h - self.crop_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if "label" in sample:
            mask = sample["label"]
            mask = mask.resize((ow, oh), Image.NEAREST)
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            return {"image": img, "label": mask, "name": sample["name"]}

        return {"image": img, "name": sample["name"]}


class Rescale(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        assert img.size == mask.size

        shape = (int(img.size[0] * self.ratio), int(img.size[1] * self.ratio))

        img = img.resize(shape, Image.BILINEAR)

        return_dict = {"image": img, "name": sample["name"]}
        if "label" in sample:
            mask = sample["label"]
            mask = mask.resize(shape, Image.NEAREST)
            # return {"image": img, "label": mask, "name": sample["name"]}
            return_dict.update({"label": mask})
        if "lanes" in sample:
            lanes = sample["lanes"]
            lanes = lanes.resize(shape, Image.NEAREST)
            return_dict.update({"lanes": lanes})
        return return_dict


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        if "label" in sample:
            mask = sample["label"]
            mask = mask.resize(self.size, Image.NEAREST)
            return {"image": img, "label": mask, "name": sample["name"]}
        return {"image": img, "name": sample["name"]}


def transform_compose_test(sample):
    composed_transforms = transforms.Compose(
        [RandomCropRect(50), Normalize(), ToTensor(),]
    )
    return composed_transforms(sample)


def test_transforms():
    image = np.ones([128, 128, 3]).astype(np.uint8)
    image = Image.fromarray(image)
    target = np.ones_like(image).astype(np.uint8)
    target = Image.fromarray(target)
    target_2 = np.ones([128, 128]).astype(np.uint8)
    target_2 = Image.fromarray(target_2)
    output_data = transform_compose_test(
        {"image": image, "targets": {"target_1": target, "target_2": target_2}}
    )
    print(output_data)


if __name__ == "__main__":
    test_transforms()
