# SOURCES FOR MOST AUGMENTATION METHODS AND PARTS OF THE CODE:
#   https://github.com/nisheethjaiswal/Data-Augmentation-for-Object-Detection/blob/main/Image_augmentation.ipynb
#   https://github.com/imjeffhi4/pokemon-classifier/blob/main/data_collection/augment_data.ipynb

import random
from PIL import ImageOps, ImageFilter
from PIL import Image
from typing import Tuple, List
import numpy as np
import cv2
from torchvision.transforms import functional
import box_transformations


def mirror(image: Image, boxes: List[Tuple]) -> Tuple[Image, List]:
    """
    Mirrors the image horizontally.

    :param image: image to modify
    :param boxes: box annotations to transform identically to image modification
    :return: modified image
    """
    boxes_transformed = [box_transformations.mirror(bbox=b, iw=image.size[0]) for b in boxes]
    return ImageOps.mirror(image), boxes_transformed


def flip(image: Image, boxes: List[Tuple]) -> Tuple[Image, List]:
    """
    Flips the image vertically.

    :param image: image to modify
    :param boxes: box annotations to transform identically to image modification
    :return: modified image
    """
    boxes_transformed = [box_transformations.flip(bbox=b, ih=image.size[1]) for b in boxes]
    return ImageOps.flip(image), boxes_transformed


def rotate(image: Image, boxes: List[Tuple], min_deg: int = 10, max_deg: int = 30) -> Tuple[Image, List]:
    """
    Adds a randomly picked rotation effect of up to max_deg degrees clockwise/counterclockwise.

    :param image: image to modify
    :param boxes: box annotations to transform identically to image modification
    :param min_deg: minimum degree to rotate the image by
    :param max_deg: maximum degree to rotate the image by
    :return: modified image
    """
    degree = random.choice(list(range(min_deg, max_deg)) + list(range(360 - min_deg, 360 - max_deg)))
    w, h = image.size
    boxes_transformed = [box_transformations.rotate(bbox=b, iw=w, ih=h, degree=degree) for b in boxes]
    return image.rotate(degree), boxes_transformed


def blur(image: Image, boxes: List[Tuple], radius_range: Tuple = (1, 2)) -> Tuple[Image, List]:
    """
    Adds a Gaussian Blur effect to the image using a randomly picked Standard deviation for the Gaussian kernel.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param radius_range: tuple of (lower limit, upper limit) of the Gaussian kernel Standard deviation to pick from
    :return: modified image
    """
    radius = random.randint(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius=radius)), boxes


def quantizing(image: Image, boxes: List[Tuple], rand_range: Tuple = (5, 5)) -> Tuple[Image, List]:
    """
    Adds a Quantizing effect to the image (reducing number of colors in image by a randomly picked value).

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param rand_range: tuple of (lower limit, upper limit) of the number of colors to reduce the image by to pick from
    :return: modified image
    """
    shift_amount = random.randint(*rand_range)
    red = (np.asarray(image)[:, :, 0] >> shift_amount) << shift_amount
    green = (np.asarray(image)[:, :, 1] >> shift_amount) << shift_amount
    blue = (np.asarray(image)[:, :, 2] >> shift_amount) << shift_amount
    return Image.fromarray(np.stack((red, green, blue), axis=2)), boxes


def noise(image: Image, boxes: List[Tuple], rand_range: Tuple = (0.6, 0.8)) -> Tuple[Image, List]:
    """
    Adds a Gaussian Noise effect to the image using a randomly picked Standard deviation and mean 0.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param rand_range: range to randomly pick standard deviation from
    :return: modified image
    """
    rand_decimal = random.random() * (rand_range[1] - rand_range[0]) + rand_range[0]
    image = np.array(image)
    gaussian = np.random.normal(0, rand_decimal, image.size)
    gaussian = gaussian.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    return Image.fromarray(cv2.add(image, gaussian)), boxes


def crop(image: Image, boxes: List[Tuple], left=(15, 60), right=(15, 60), top=(15, 60), bottom=(15, 60)) -> Tuple[Image, List]:
    """
    Adds a random cropping effect to all four sides of the image using the specified cropping ranges.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param left: range to randomly pick crop amount for left side from
    :param right: range to randomly pick crop amount for right side from
    :param top: range to randomly pick crop amount for top side from
    :param bottom: range to randomly pick crop amount for bottom side from
    :return: modified image
    """
    w, h = image.size
    l_crop = random.randint(left[0], left[1])
    r_crop = random.randint(right[0], right[1])
    t_crop = random.randint(top[0], top[1])
    b_crop = random.randint(bottom[0], bottom[1])
    boxes_ = [box_transformations.crop(box, l_crop, r_crop, t_crop, b_crop, w, h) for box in boxes]
    return image.crop((l_crop, t_crop, w - r_crop, h - b_crop)).resize((w, h)), boxes_


def increase_contrast(image: Image, boxes: List[Tuple], c: float = None) -> Tuple[Image, List]:
    """
    Multiplies the contrast by factor c.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param c: factor to multiply contrast by
    :return: modified image
    """
    if c is None:
        c = random.uniform(3.0, 5.0)
    return functional.adjust_contrast(image, c), boxes


def reduce_contrast(image: Image, boxes: List[Tuple], c: float = None) -> Tuple[Image, List]:
    """
    Multiplies the contrast by factor c.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param c: factor to multiply contrast by
    :return: modified image
    """
    if c is None:
        c = random.uniform(0.3, 0.5)
    return functional.adjust_contrast(image, c), boxes


def increase_brightness(image: Image, boxes: List[Tuple], b: float = None) -> Tuple[Image, List]:
    """
    Multiplies the brightness by factor b.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param b: factor to brightness contrast by
    :return: modified image
    """
    if b is None:
        b = random.uniform(1.25, 1.75)
    return functional.adjust_brightness(image, b), boxes


def reduce_brightness(image: Image, boxes: List[Tuple], b: float = None) -> Tuple[Image, List]:
    """
    Multiplies the brightness by factor b.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param b: factor to brightness contrast by
    :return: modified image
    """
    if b is None:
        b = random.uniform(0.5, 0.75)
    return functional.adjust_brightness(image, b), boxes


def increase_saturation(image: Image, boxes: List[Tuple], s: float = None) -> Tuple[Image, List]:
    """
    Multiplies the saturation by factor s.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param s: factor to multiply color saturation by
    :return: modified image
    """
    if s is None:
        s = random.uniform(1.25, 1.75)
    return functional.adjust_saturation(image, s), boxes


def reduce_saturation(image: Image, boxes: List[Tuple], s: float = None) -> Tuple[Image, List]:
    """
    Multiplies the saturation by factor s.

    :param image: image to modify
    :param boxes: box annotations to return unchanged
    :param s: factor to multiply color saturation by
    :return: modified image
    """
    if s is None:
        s = random.uniform(0.5, 0.75)
    return functional.adjust_saturation(image, s), boxes
