import numpy as np
import os 
from detectron2.structures import BoxMode

def face_dataset_dicts(img_dir, project_dir):
    dataset_dicts = []
    json_file = os.path.join(project_dir, "annot_all", "annot_all.json")

    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, f_name in enumerate(imgs_anns.keys()):
        record = {}

        filename = os.path.join(img_dir, f_name)
        height, width = imgs_anns[f_name]["height"], imgs_anns[f_name]["width"]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = imgs_anns[f_name]["annotation"]

        objs = []
        for bbox in annos:

            xmin = bbox["xmin"]
            xmax = bbox["xmax"]
            ymin = bbox["ymin"]
            ymax = bbox["ymax"]

            poly = [
                (xmin, ymin), (xmax, ymin),
                (xmax, ymax), (xmin, ymax)
            ]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "is_crowd": 0
            }

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

def register_dataset_detectron2():

    project_dir = "../VOCdevkit2007/VOC2007"
    img_dir = "../VOCdevkit2007/VOC2007/face_dataset"

    for d in ["train", "val"]:
        DatasetCatalog.register("anime_face_" + d, lambda d=d: face_dataset_dicts(os.path.join(img_dir, d), project_dir))
        MetadataCatalog.get("anime_face_" + d).set(thing_classes=["face"])


if __name__ == "__main__":
    a = {"a":1, "b":2, "c":3} 
    for idx, val in enumerate(a.keys()):
        print(val)  
    