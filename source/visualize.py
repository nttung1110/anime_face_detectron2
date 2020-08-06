import cv2
import numpy as np
import xmltodict
import os 

path_img_data = "../VOCdevkit2007/VOC2007/JPEGImages"
path_anno_data = "../VOCdevkit2007/VOC2007/Annotations"
path_vis_output = "../VOCdevkit2007/VOC2007/Vis_bbox"

def j_p(s1, s2):
    return os.path.join(s1, s2)

def read_xml_file(xml_fn):
    s = "".join(open(xml_fn, 'r').readlines())
    dct = xmltodict.parse(s)
    return dct

def vis(img_name, annot_dict):
    img_f_p = j_p(path_img_data, img_name)
    img = cv2.imread(img_f_p)
    res_img = img

    if "object" in annot_dict["annotation"]:
        bbox_list = []
        for each_obj in annot_dict["annotation"]["object"]:
            if isinstance(each_obj, str) == False:
                bbox = each_obj["bndbox"]
                bbox = [int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])]
                bbox_list.append(bbox)
            
            else:
                bbox = annot_dict["annotation"]["object"]["bndbox"]
                bbox_list.append([int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])])

        # draw bbox
        for each_bbox in bbox_list:
            res_img = cv2.rectangle(res_img, (each_bbox[0], each_bbox[1]), (each_bbox[2], each_bbox[3]), (255, 0, 0), 2)

    res_img_f_p = j_p(path_vis_output, img_name)

    cv2.imwrite(res_img_f_p, res_img)


def main(count_limit):
    if os.path.isdir(path_vis_output) == False:
        os.mkdir(path_vis_output)

    count = 0
    # processed each xml
    for xml_file in os.listdir(path_anno_data):
        if xml_file.endswith(".xml"):
            print("Count:", count+1)
            xml_f_p = j_p(path_anno_data, xml_file)
            annot = read_xml_file(xml_f_p)

            vis(annot["annotation"]["filename"], annot)
            count+=1

            if count == count_limit:
                return

    

if __name__ == "__main__":
    main(7000)