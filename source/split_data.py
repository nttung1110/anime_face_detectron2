import numpy as np
import json
import cv2 
import os 
import shutil

def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def transfer(src_path, des_path, list_id, type_data, all_annot):
    des_path_convert = os.path.join(des_path, type_data)

    makedir(des_path_convert)

    for id in list_id:
        if id + ".jpg" not in all_annot: # image do not have objects=>reject
            continue
        print("Transfer", id)
        src_path_img = os.path.join(src_path, id+".jpg")
        des_path_img = os.path.join(des_path_convert, id+".jpg")
        
        dest = shutil.copy(src_path_img, des_path_img)

def create_annotate(type_data, all_annot, list_id, out_path_annot):
    full_path_out = os.path.join(out_path_annot, type_data+".json")
    res_dict = {}

    for id_file in list_id:
        id_with_tail = id_file + ".jpg"
        if id_with_tail not in all_annot: # image do not have objects=>reject
            continue
        res_dict[id_with_tail] = all_annot[id_with_tail]

    with open(full_path_out, "w") as jsonFile:
        json.dump(res_dict, jsonFile, indent=4, sort_keys=True)


def split():
    '''
        Split both image and annotation
    '''
    # image path
    src_img_path = "../../VOCdevkit2007/VOC2007/JPEGImages"
    des_img_path = "../data/face_dataset"

    # annotation path
    source_annot_all_file = "../data/annot_all/annot_all.json"
    l_train_val_id_path = "../list_split_id/trainval.txt"
    l_test_id_path = "../list_split_id/test.txt"
    out_path_annot = "../data/annot_all"



    with open(l_train_val_id_path, "r") as f:
        list_tv = f.read().split()

    with open(l_test_id_path, "r") as f:
        list_test = f.read().split()

    train_val_ratio = 0.7

    list_train = list_tv[:int(train_val_ratio*len(list_tv))]
    list_val = list_tv[int(train_val_ratio*len(list_tv)):]

    with open(source_annot_all_file) as f:
        all_annot = json.load(f)

    for type_data, list_id in zip(["train", "val", "test"], [list_train, list_val, list_test]):
        transfer(src_img_path, des_img_path, list_id, type_data, all_annot)
        create_annotate(type_data, all_annot, list_id, out_path_annot)


if __name__ == "__main__":
    split()