# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys



from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set model
    cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="./configs/ocr/icdar2015_101_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--weights",
        default="./out_dir_r101/icdar2015_model/model_ic15_r101.pth",
        metavar="pth",
        help="the model used to inference",
    )

    parser.add_argument(
        "--input",
        default="./input_images/*.jpg",
        nargs="+",
        help="the folder of icdar2015 test images"
    )

    parser.add_argument(
        "--output",
        default="./test_icdar2015/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def compute_polygon_area(points):
    s = 0
    point_num = len(points)
    if(point_num < 3): return 0.0
    for i in range(point_num): 
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)
    

def save_result_to_txt(txt_save_path,prediction, contours):

    file = open(txt_save_path,'w')

    b_boxes = get_bboxes(contours)

    # print(b_boxes)
    for box in b_boxes:
      area = compute_polygon_area(box)
      # print(f'area: {area}')
      if area > 175:
        file.writelines(str(int(box[0][0]))+','+str(int(box[0][1]))+','+str(int(box[1][0]))+','+str(int(box[1][1]))+','
                              +str(int(box[2][0]))+','+str(int(box[2][1]))+','+str(int(box[3][0]))+','+str(int(box[3][1])))
        file.write('\r\n')


    # classes = prediction['instances'].pred_classes
    # polygons = prediction['instances'].pred_boxes

    # for i in range(len(classes)):
    #     if classes[i]==0:
    #         if len(polygons[i]) != 0:
    #             points = [polygons[i][:2], polygons[i][2:]]

                # points = []
                # for j in range(0,len(polygons[i][0]),2):
                #     points.append([polygons[i][0][j],polygons[i][0][j+1]])
                # points = np.array(points)
                # area = compute_polygon_area(points)
                # rect = cv2.minAreaRect(points)
                # box = cv2.boxPoints(rect)

                # if area > 175:
                #     file.writelines(str(int(box[0][0]))+','+str(int(box[0][1]))+','+str(int(box[1][0]))+','+str(int(box[1][1]))+','
                #               +str(int(box[2][0]))+','+str(int(box[2][1]))+','+str(int(box[3][0]))+','+str(int(box[3][1])))
                #     file.write('\r\n')

    file.close()



def get_bboxes(contours):
    cnts = list()
    for cont in contours:
        rect = cv2.minAreaRect(cont)
    
        if min(rect[1][0], rect[1][1]) <= 5:
            continue
        points = cv2.boxPoints(rect)
        points = np.intp(points)
        cnts.append(points)
    return np.array(cnts)


if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = setup_cfg(args)
    detection_demo = VisualizationDemo(cfg)

    test_images_path = args.input
    output_path = args.output

    start_time_all = time.time()
    img_count = 0
    for i in glob.glob(test_images_path[0]):

        if img_count == 20:
          break

        print(i)
        img_name = os.path.basename(i)
        img_save_path = output_path + img_name.split('.')[0] + '.jpg'
        img = cv2.imread(i)
        start_time = time.time()

        prediction, vis_output = detection_demo.run_on_image(img)
        print(f"prediction: {prediction['instances'].pred_masks}")
        # print(f'vis_output: {vis_output.get_image()}')
        # print(f'Image b_boxes: {type(polygons)} ---> {len(polygons)}')
        vis_output.save(img_save_path)

        

        # "outputs" is the inference output in the format described here - https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        # Extract the contour of each predicted mask and save it in a list
        contours = []
        for pred_mask in prediction['instances'].pred_masks:
            # pred_mask is of type torch.Tensor, and the values are boolean (True, False)
            # Convert it to a 8-bit numpy array, which can then be used to find contours
            mask = np.array(pred_mask.tolist(), dtype=np.uint8)
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            contours.append(contour[0]) # contour is a tuple (OpenCV 4.5.2), so take the first element which is the array of contour points

        # image_with_overlaid_predictions = img.copy()

        # for contour in contours:
        #     cv2.drawContours(image_with_overlaid_predictions, [contour], -1, (0,255,0), 1)

        # cv2.imwrite(img_save_path.replace('.jpg', '_contours.png'), image_with_overlaid_predictions)

        
        


        txt_save_path = output_path + 'res_' + img_name.split('.')[0] + '.txt'
        save_result_to_txt(txt_save_path,prediction, contours)

        print("Time: {:.2f} s / img".format(time.time() - start_time))
        img_count += 1
    print("Average Time: {:.2f} s /img".format((time.time() - start_time_all) / img_count))


