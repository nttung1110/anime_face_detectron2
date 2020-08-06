
from detectron2.structures import BoxMode

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

project_dir = "../data"
img_dir = "../data/face_dataset"

def face_dataset_dicts():
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

def test():

    dataset_dicts = face_dataset_dicts()
    anime_face_metadata = MetadataCatalog.get("anime_face_train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=anime_face_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite(d["file_name"]+".jpg", out.get_image()[:, :, ::-1])

def register_dataset_detectron2():

    for d in ["train", "val"]:
        DatasetCatalog.register("anime_face_" + d, lambda d=d: face_dataset_dicts()
        MetadataCatalog.get("anime_face_" + d).set(thing_classes=["face"])

    test()


if __name__ == "__main__":
    register_dataset_detectron2()
    