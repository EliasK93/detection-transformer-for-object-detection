import math
from typing import Tuple, List


def get_bbox_from_cx_cy_w_h(cx: float, cy: float, rw: float, rh: float, iw: int, ih: int) -> List[int]:
    """
    Helper method to convert bounding box format from
    (procentual_centerX, procentual_centerY, procentual_box_width, procentual_box_height)
    to
    (absolute_left_x, absolute_top_y, absolute_box_width, absolute_box_height)

    :param cx: x coordinate of center of bounding box as share of image size (between 0.0 and 1.0)
    :param cy: y coordinate of center of bounding box as share of image size (between 0.0 and 1.0)
    :param rw: width of bounding box as share of image size (between 0.0 and 1.0)
    :param rh: height of bounding box as share of image size (between 0.0 and 1.0)
    :param iw: image width in pixels
    :param ih: image height in pixels
    :return: bounding box in format [absolute_left_x, absolute_top_y, absolute_box_width, absolute_box_height]
    """
    w = int(iw * rw)
    h = int(ih * rh)
    x1 = int(iw * cx) - w // 2
    y1 = int(ih * cy) - h // 2
    return [x1, y1, w, h]


def get_bbox_from_x1_y1_x2_y2(ann: Tuple) -> List[int]:
    """
    Helper method to convert bounding box format from
    (absolute_left_x, absolute_top_y, absolute_right_x, absolute_bottom_y)
    to
    (absolute_left_x, absolute_top_y, absolute_box_width, absolute_box_height)

    :param ann: box tuple (absolute_left_x, absolute_top_y, absolute_right_x, absolute_bottom_y)
    :return: bounding box in format [absolute_left_x, absolute_top_y, absolute_box_width, absolute_box_height]
    """
    x1, y1, x2, y2 = ann
    return [x1, y1, x2-x1, y2-y1]


def rotate(bbox: Tuple[int, int, int, int], iw: int, ih: int, degree: int) -> Tuple[int, int, int, int]:
    """
    Rotates the annotated bounding box (counterclockwise) using the given degree.

    :param bbox: tuple(xmin, ymin, xmax, ymax)
    :param iw: image width
    :param ih: image height
    :param degree: degree
    :return: transformed bbox
    """
    # center of image (rotation point)
    cx = iw/2
    cy = ih/2

    # old center of box
    box_x1, box_y1, box_x2, box_y2 = bbox
    box_w = box_x2 - box_x1
    box_h = box_y2 - box_y1
    x = box_x2 - box_w/2
    y = box_y2 - box_h/2

    # new center of box
    x_new = math.cos(math.radians(-degree))*(x-cx)-math.sin(math.radians(-degree))*(y-cy)+cx
    y_new = math.sin(math.radians(-degree))*(x-cx)+math.cos(math.radians(-degree))*(y-cy)+cy

    # new x1, y1
    box_x1_new = min(max(int(x_new - box_w/2), 0), iw)
    box_y1_new = min(max(int(y_new - box_h/2), 0), ih)
    box_x2_new = min(max(int(x_new + box_w/2), 0), iw)
    box_y2_new = min(max(int(y_new + box_h/2), 0), ih)

    return box_x1_new, box_y1_new, box_x2_new, box_y2_new


def mirror(bbox: Tuple[int, int, int, int], iw: int) -> Tuple[int, int, int, int]:
    """
    Mirrors the annotated bounding box horizontally.

    :param bbox: tuple(xmin, ymin, xmax, ymax)
    :param iw: image width
    """
    xmin, ymin, xmax, ymax = bbox
    return iw - xmin, ymin, iw - xmax, ymax


def flip(bbox: Tuple[int, int, int, int], ih: int) -> Tuple[int, int, int, int]:
    """
    Mirrors the annotated bounding box vertically.

    :param bbox: tuple(xmin, ymin, xmax, ymax)
    :param ih: image height
    """
    xmin, ymin, xmax, ymax = bbox
    return xmin, ih - ymin, xmax, ih - ymax


def crop(bbox: int, left: int, right: int, top: int, bottom: int, iw: int, ih: int) -> Tuple[int, int, int, int]:
    """
    Matches the annotation bounding boxes to the cropped image by first adjusting for the cuts and then
    adjusting for the image stretching.

    :param bbox: tuple(xmin, ymin, xmax, ymax)
    :param left: width cut from left side
    :param right: width cut from right side
    :param top: height cut from top side
    :param bottom: height cut from bottom side
    :param iw: image width
    :param ih: image height
    :return: transformed bbox
    """

    # adjust for cuts
    box_x1, box_y1, box_x2, box_y2 = bbox
    box_x1 = box_x1 - left
    box_x2 = box_x2 - left
    box_y1 = box_y1 - top
    box_y2 = box_y2 - top

    # adjust for stretching
    x_multiplier = iw/(iw-left-right)
    y_multiplier = ih/(ih-top-bottom)
    box_x1 = max(0, int(round(box_x1 * x_multiplier, 0)))
    box_x2 = min(iw, int(round(box_x2 * x_multiplier, 0)))
    box_y1 = max(0, int(round(box_y1 * y_multiplier, 0)))
    box_y2 = min(ih, int(round(box_y2 * y_multiplier, 0)))

    return box_x1, box_y1, box_x2, box_y2
