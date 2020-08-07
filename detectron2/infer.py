import os
import train_anime_face
from train_anime_face import my_cfg
import cv2
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor

from dataset import register_dataset_detectron2, face_dataset_dicts
from detectron2.data import MetadataCatalog, DatasetCatalog


from detectron2.utils.visualizer import ColorMode
def infer(predictor):
    img_dir = "../data/face_dataset/test"
    annot_dir = "../data/annot_all/test.json"

    dataset_dicts = face_dataset_dicts(img_dir, annot_dir)
    anime_face_metadata = MetadataCatalog.get("anime_face_test")

    # for d in random.sample(dataset_dicts, 3):    
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1],
    #                 metadata=anime_face_metadata, 
    #                 scale=0.5, 
    #                 instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #     )
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imwrite("test_result/"+d["file_name"].split("/")[-1], out.get_image()[:, :, ::-1])

    # test HOR data hor02_056
    for file_name in os.listdir("test_data"):
        if file_name.endswith(".jpg"):
            full_path = os.path.join("test_data", file_name)
            img = cv2.imread(full_path)
            outputs = predictor(img)
            print(outputs["instances"])
            v = Visualizer(img[:, :, ::-1],
                            metadata=anime_face_metadata, 
                            scale=0.5, 
                            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite("test_result/"+file_name, out.get_image()[:, :, ::-1])

def main():
    # register first
    register_dataset_detectron2()

    cfg = my_cfg()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.DATASETS.TEST = ("anime_face_test", )
    predictor = DefaultPredictor(cfg)

    infer(predictor)

if __name__ == "__main__":
    main()