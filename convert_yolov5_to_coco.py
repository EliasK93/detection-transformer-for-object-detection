import json
import os
import cv2
import box_transformations


if __name__ == '__main__':

    YOLOv5_data_path = "data_yolov5/lunar/craters"

    # define category names, supercategory name and category mapping
    supercategory = "craters"
    categories = [{"id": 0, "name": supercategory, "supercategory": "none"}]
    categories += [{"id": i, "name": c["id"], "supercategory": supercategory}
                   for i, c in enumerate([{"id": "crater", "description": "crater"}], start=1)]
    cat2id = {c["name"]: c["id"] for c in categories}

    for set_type in ["train", "valid", "test"]:
        # path to load images from
        path = f"{YOLOv5_data_path}/{set_type}/images"

        # load images, add some fields required for COCO format, including width and heights
        image_files = os.listdir(path)
        widths_and_heights = [cv2.imread(f"{YOLOv5_data_path}/{set_type}/images/" + loc).shape[:2] for loc in image_files]
        images = [{"id": i, "license": 1, "file_name": loc, "height": h, "width": w,
                   "date_captured": "2022-01-01T00:00:00+00:00"} for i, (loc, (h, w)) in
                  enumerate(zip(image_files, widths_and_heights))]

        # load annotations in YOLOv5 format, put them in list nested by images
        label_files = os.listdir(f"{YOLOv5_data_path}/{set_type}/labels/")
        annotations_nested_by_images = []
        for file in label_files:
            with open(f"{YOLOv5_data_path}/{set_type}/labels/" + file, encoding="utf-8") as f:
                lines = [[tuple([float(i) for i in line.rstrip().split()])] for line in f]
                annotations_nested_by_images.append([[{"centerX": b, "centerY": c, "width": d,
                                                      "height": e, "classification": "crater"}
                                                      for a, b, c, d, e in line] for line in lines])

        # format and add annotation fields required for COCO format, including different bbox notation
        annotations = []
        for img_id, (annotation_list, image_dict) in enumerate(zip(annotations_nested_by_images, images)):
            for annotation in annotation_list:
                assert len(annotation) == 1
                annotation = annotation[0]
                annotations.append({
                    "id": len(annotations),
                    "image_id": img_id,
                    "category_id": cat2id[annotation["classification"]],
                    "bbox": box_transformations.get_bbox_from_cx_cy_w_h(
                        annotation["centerX"], annotation["centerY"], annotation["width"],
                        annotation["height"], image_dict["width"], image_dict["height"]),
                    "area": None,
                    "segmentation": [],
                    "iscrowd": 0
                })

        # load layout file to get the other COCO format fields
        with open("layout.coco.json") as f:
            layout = json.load(f)

        # insert the formatted categories, images and annotations
        layout["categories"] = categories
        layout["images"] = images
        layout["annotations"] = annotations

        # write to annotation file
        if not os.path.exists("data_coco/" + supercategory + "/" + set_type):
            os.makedirs("data_coco/" + supercategory + "/" + set_type)
        with open("data_coco/" + supercategory + "/" + set_type + "/_annotations.coco.json", "w") as f:
            json.dump(layout, f)
