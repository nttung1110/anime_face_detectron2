import os
import train_anime_face
from train_anime_face import my_cfg
import cv2
import PIL
import numpy as np
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor

from dataset import register_dataset_detectron2, face_dataset_dicts
from detectron2.data import MetadataCatalog, DatasetCatalog


from detectron2.utils.visualizer import ColorMode

def visualize_bbox(img, bbox):
    if bbox != None:
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
    return img

def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def infer_folder(predictor, source_folder, des_folder):
    print("Processing folder ", source_folder)

    # test HOR data hor02_056
    makedir(des_folder)
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".tga") or file_name.endswith(".jpg"):
            full_path = os.path.join(source_folder, file_name)
            img = cv2.cvtColor(np.array(PIL.Image.open(full_path)), cv2.COLOR_RGB2BGR)
            outputs = predictor(img)

            num_instance = len(outputs["instances"])
            out_img_path = os.path.join(des_folder, file_name[:-4]+".jpg")
            if num_instance == 0:
                # visualize without bbox
                img = visualize_bbox(img, None)
                
            else:    
                bbox = outputs["instances"].get("pred_boxes").tensor[0]
                img = visualize_bbox(img, bbox)
            cv2.imwrite(out_img_path, img)

            

def main():
    # register first

    cfg = my_cfg()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.DATASETS.TEST = ("anime_face_test", )
    predictor = DefaultPredictor(cfg)

    # # HOR01
    path_HOR01 = "../HOR_DATA/HOR01_Full_Formated"
    out_path_HOR01 = "../HOR_DATA/result_HOR01"
    for sub_folder in os.listdir(path_HOR01):
        source_path = os.path.join(path_HOR01, sub_folder, "color")
        if not os.path.isdir(source_path):
            continue
        des_path = os.path.join(out_path_HOR01, sub_folder)

        infer_folder(predictor, source_path, des_path)

    # HOR02
    path_HOR02 = "../HOR_DATA/HOR02_Full_Formated"
    out_path_HOR02 = "../HOR_DATA/result_HOR02"

    for sub_folder in os.listdir(path_HOR02):
        source_path = os.path.join(path_HOR02, sub_folder, "color")
        if not os.path.isdir(source_path):
            continue
        des_path = os.path.join(out_path_HOR02, sub_folder)

        infer_folder(predictor, source_path, des_path)
    # infer_folder(predictor, "test_data", "test_result")

if __name__ == "__main__":
    main()