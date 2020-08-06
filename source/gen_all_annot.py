import cv2
import numpy as np
import json
import xmltodict
import os 

path_img_data = "../VOCdevkit2007/VOC2007/JPEGImages"

path_anno_data = "../VOCdevkit2007/VOC2007/Annotations"
path_all_anno_output = "../VOCdevkit2007/VOC2007/annot_all"

def j_p(s1, s2):
    return os.path.join(s1, s2)

def read_xml_file(xml_fn):
    s = "".join(open(xml_fn, 'r').readlines())
    dct = xmltodict.parse(s)
    return dct

def processed(img_name, annot_dict):
    img_f_p = j_p(path_img_data, img_name)
    img = cv2.imread(img_f_p)
    res_annot = {}

    res_annot["height"] = img.shape[0]
    res_annot["width"] = img.shape[1]
    
    list_annot = [] # each element is a dict representing for each object annotation

    if "object" in annot_dict["annotation"]:
        bbox_list = []

        for each_obj in annot_dict["annotation"]["object"]:
            each_annot = {}

            if isinstance(each_obj, str) == False:
                bbox = each_obj["bndbox"]
            
            else:
                bbox = annot_dict["annotation"]["object"]["bndbox"]

            each_annot["xmin"] = int(bbox["xmin"])
            each_annot["ymin"] = int(bbox["ymin"])
            each_annot["xmax"] = int(bbox["xmax"])
            each_annot["ymax"] = int(bbox["ymax"])

            list_annot.append(each_annot)
        
        res_annot["annotation"] = list_annot

        return res_annot
    
    return None # no face in image



def main(count_limit):
    if os.path.isdir(path_all_anno_output) == False:
        os.mkdir(path_all_anno_output)

    count = 0
    # processed each xml
    all_annot = {}

    l_train_val_id_path = "../list_split_id/trainval.txt"
    l_test_id_path = "../list_split_id/test.txt"

    with open(l_train_val_id_path, "r") as f:
        list_tv = f.read().split()

    with open(l_test_id_path, "r") as f:
        list_test = f.read().split()

    for xml_file in os.listdir(path_anno_data):
        if xml_file.endswith(".xml"):
            print("Count:", count+1)
            xml_f_p = j_p(path_anno_data, xml_file)
            annot = read_xml_file(xml_f_p)

            f_name = annot["annotation"]["filename"]


            each_processed_annot = processed(f_name, annot)
            
            if each_processed_annot is None:
                continue 

            all_annot[f_name] = each_processed_annot
            count+=1
            
            # if count == count_limit:
            #     break

    with open(os.path.join(path_all_anno_output, "annot_all.json"), "w") as jsonFile:
        json.dump(all_annot, jsonFile, indent=4, sort_keys=True)


if __name__ == "__main__":
    main(10)

