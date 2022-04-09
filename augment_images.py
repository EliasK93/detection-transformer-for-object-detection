import json
import os
import random
from typing import Tuple, Dict, List
import cv2
from PIL import Image
import box_transformations
import image_transformations
import matplotlib.pyplot as plt
from tqdm import tqdm


def augment(pil_img: Image, boxes: List[Tuple], aug_type: str) -> Tuple[Image, List]:
    """
    Applies the selected augmentation method to the provided image and boxes.

    :param pil_img: image to augment
    :param boxes: bounding boxes before augmentation
    :param aug_type: augmentation type
    :return:
    """
    augmention_functions = {
        "mirror": image_transformations.mirror,
        "flip": image_transformations.flip,
        "rotate": image_transformations.rotate,
        "blur": image_transformations.blur,
        "quantizing": image_transformations.quantizing,
        "noise": image_transformations.noise,
        "crop": image_transformations.crop,
        "increase_brightness": image_transformations.increase_brightness,
        "reduce_brightness": image_transformations.reduce_brightness,
        "increase_contrast": image_transformations.increase_contrast,
        "reduce_contrast": image_transformations.reduce_contrast,
        "increase_saturation": image_transformations.increase_saturation,
        "reduce_saturation": image_transformations.reduce_saturation
    }
    return augmention_functions[aug_type](pil_img, boxes)


def plot_results(pil_img: Image, boxes: List, i: int, aug: str):
    """
    Helper method to plot an image and it's annotated boxes.

    :param pil_img: image to show
    :param boxes: annotated bounding boxes [(absolute_left_x, absolute_top_y, absolute_right_x, absolute_bottom_y), ...]
    :param i: image index
    :param aug: augmentation type
    """
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] * 100
    plt.imshow(pil_img)
    ax = plt.gca()
    for (xmin, ymin, xmax, ymax), c in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
    plt.axis('off')
    plt.savefig(f"SAMPLES/{i}_{aug}.png", bbox_inches='tight', pad_inches=0)
    plt.close()


def load_image_dicts(set_type: str) -> Dict[str, Tuple]:
    """
    Load annotations in COCO format from file and return in a simplified dict format.

    :param set_type: set type (train, valid or test)
    :return: images and annotations in a simplified dict format: {image1_file_name: [bbox1, bbox2, ...], ...}
    """
    with open(f"data_coco/craters/{set_type}/_annotations.coco.json") as f:
        content = json.load(f)
    return {d["file_name"]: [d_["bbox"] for d_ in content["annotations"] if d_["image_id"] == d["id"]]
            for d in content["images"]}


def create_and_save_augmented(image_dict: Dict[str, Tuple], set_type: str, plot: bool = False) -> List[Tuple]:
    """
    Load images in image_dict, augment images and annotations and return in a simplified tuple list format.

    :param image_dict: images and annotations in simplified dict format {image1_file_name: [bbox1, bbox2, ...], ...}
    :param set_type: set type (train, valid or test)
    :param plot: whether to plot each image + annotation and show it using matplotlib
    :return: images and annotations in a simplified tuple list format:
             [(PIL_image1_original, [bbox1_orig, bbox2_orig, ...]), (PIL_image1_aug1, [bbox1_aug1, bbox2_aug1, ...])]
    """
    augmented_images = []
    for i, (file_name, boxes) in tqdm(enumerate(image_dict.items()), desc="Augmentation: " + set_type):
        boxes = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in boxes]
        # save original under unified name
        original_image = Image.open(f"data_coco/craters/{set_type}/" + file_name)
        original_image.save(f"data_augmented/{set_type}/{i}_0.jpg")
        augmented_images.append((f"{i}_0.jpg", boxes))
        if set_type == "train":
            if plot:
                plot_results(original_image, boxes, i, "original")
            for j, aug in enumerate(["mirror", "flip", "rotate", "blur", "quantizing", "noise", "crop",
                                     "increase_brightness", "reduce_brightness", "increase_contrast",
                                     "reduce_contrast", "increase_saturation", "reduce_saturation"], start=1):
                if plot:
                    print(aug)
                augmented_image, augmented_boxes = augment(original_image, boxes, aug_type=aug)
                augmented_image.save(f"data_augmented/{set_type}/{i}_{j}.jpg")
                augmented_images.append((f"{i}_{j}.jpg", augmented_boxes))
                if plot:
                    plot_results(augmented_image, augmented_boxes, i, aug)
    return augmented_images


def create_reformatted_image_dict(augmented_images: List[Tuple], set_type: str) -> List[Dict]:
    """
    Converts images from simplified tuple list format to image dict list in COCO format.

    :param augmented_images: images and annotations in a simplified tuple list format:
                             [(PIL_image1_original, [bbox1_orig, bbox2_orig, ...]),
                             (PIL_image1_aug1, [bbox1_aug1, bbox2_aug1, ...])]
    :param set_type: set type (train, valid or test)
    :return: image dict list in coco format
    """
    widths_and_heights = [cv2.imread(f"data_augmented/{set_type}/{loc[0]}").shape[:2] for loc in augmented_images]
    return [{"id": i, "license": 1, "file_name": loc, "height": h, "width": w,
             "date_captured": "2022-01-01T00:00:00+00:00"} for i, (loc, (h, w)) in
            enumerate(zip([a[0] for a in augmented_images], widths_and_heights))]


def create_reformatted_annotation_dict(augmented_images: List[Tuple], image_dict_list: List[Dict]) -> List[Dict]:
    """
    Converts annotations from simplified tuple list format to annotation dict list in COCO format.

    :param augmented_images: images and annotations in a simplified tuple list format:
                             [(PIL_image1_original, [bbox1_orig, bbox2_orig, ...]),
                             (PIL_image1_aug1, [bbox1_aug1, bbox2_aug1, ...])]
    :param image_dict_list: image dict list in coco format
    :return: annotation dict list in coco format
    """
    annotations = []
    for img_id, (ann_list, image_dict) in enumerate(zip([a[1] for a in augmented_images], image_dict_list)):
        for ann in ann_list:
            a = {
                "id": len(annotations),
                "image_id": img_id,
                "category_id": 1,
                "bbox": box_transformations.get_bbox_from_x1_y1_x2_y2(ann),
                "area": None,
                "segmentation": [],
                "iscrowd": 0
            }
            annotations.append(a)
    return annotations


def wrap_images_and_annotations_in_layout(image_dict_list: List[Dict], annotation_dict_list: List[Dict]) -> Dict:
    """
    Loads a layout wrapper file for the COCO format, puts the image dict list and annotation dict list in COCO
    format in it and returns the final dataset in COCO format as a dict.

    :param image_dict_list: image dict list in coco format
    :param annotation_dict_list: annotation dict list in coco format
    :return: final dataset in coco format as a dictionary
    """
    with open("layout.coco.json") as f:
        layout = json.load(f)
    supercategory = "craters"
    categories = [{"id": 0, "name": supercategory, "supercategory": "none"}]
    categories += [{"id": i, "name": l["id"], "supercategory": supercategory}
                   for i, l in enumerate([{"id": "crater", "description": "crater"}], start=1)]
    layout["categories"] = categories
    layout["images"] = image_dict_list
    layout["annotations"] = annotation_dict_list
    return layout


def run_augmentation(seed: int = 1, plot: bool = False):
    """
    Main method to create augmented dataset.

    :param seed: random seed to use for augmentation
    :param plot: Plots all created augmented images if True
    """
    random.seed(seed)

    for set_type in ["train", "valid", "test"]:

        # load images and annotations
        images = load_image_dicts(set_type)

        # create directory
        if not os.path.exists(f"data_augmented/{set_type}/"):
            os.makedirs(f"data_augmented/{set_type}/")

        # augment images and annotations
        augmented_dict = create_and_save_augmented(image_dict=images, set_type=set_type, plot=plot)

        # convert to image dict list in coco format
        image_dict_list = create_reformatted_image_dict(augmented_dict, set_type)

        # convert to annotation dict list in coco format
        annotation_dict_list = create_reformatted_annotation_dict(augmented_dict, image_dict_list)

        # wrap in coco layout file
        wrapped_in_layout = wrap_images_and_annotations_in_layout(image_dict_list, annotation_dict_list)

        # write to file
        with open(f"data_augmented/{set_type}/_annotations.coco.json", "w") as f:
            json.dump(wrapped_in_layout, f)


if __name__ == '__main__':
    run_augmentation(seed=1, plot=False)
