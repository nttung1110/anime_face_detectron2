import numpy as np
import cv2 
import os 
import shutil

def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def transfer(src_path, des_path, list_id, type_data):
    des_path_convert = os.path.join(des_path, type_data)

    makedir(des_path_convert)

    for id in list_id:
        print("Transfer", id)
        src_path_img = os.path.join(src_path, id+".jpg")
        des_path_img = os.path.join(des_path_convert, id+".jpg")
        
        dest = shutil.copy(src_path_img, des_path_img)

def split():

    src_img_path = "../VOCdevkit2007/VOC2007/JPEGImages"
    des_img_path = "../VOCdevkit2007/VOC2007/face_dataset"
    l_train_val_id_path = "../VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
    l_test_id_path = "../VOCdevkit2007/VOC2007/ImageSets/Main/test.txt"

    with open(l_train_val_id_path, "r") as f:
        list_tv = f.read().split()

    with open(l_test_id_path, "r") as f:
        list_test = f.read().split()

    train_val_ratio = 0.7

    list_train = list_tv[:int(train_val_ratio*len(list_tv))]
    list_val = list_tv[int(train_val_ratio*len(list_tv)):]

    for type_data, list_id in zip(["train", "val", "test"], [list_train, list_val, list_test]):
        transfer(src_img_path, des_img_path, list_id, type_data)


if __name__ == "__main__":
    split()