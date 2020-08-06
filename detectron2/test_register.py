import random, cv2
from detectron2.utils.visualizer import Visualizer


def test():
    dataset_dicts = get_balloon_dicts("balloon/train")
    anime_face_metadata = MetadataCatalog.get("anime_face_train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=anime_face_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite(d["file_name"]+".jpg", out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    test()