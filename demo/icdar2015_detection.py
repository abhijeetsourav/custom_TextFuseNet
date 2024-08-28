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
import csv


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
    

def save_result_to_csv(csv_save_path,prediction, b_boxes):

    classes = prediction['instances'].pred_classes
    scores = prediction['instances'].scores.tolist()    

    with open(csv_save_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'score'])



        # print(b_boxes)
        for i, box in enumerate(b_boxes):
            if classes[i] != 0:
                break
            area = compute_polygon_area(box)
            # print(f'area: {area}')
            if area > 175:
                csvwriter.writerows(int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1]), int(box[2][0]), int(box[2][1]), int(box[3][0]), int(box[3][1]), int(scores[100] * 100) / 100.0 )
                # file.writelines(str(int(box[0][0]))+','+str(int(box[0][1]))+','+str(int(box[1][0]))+','+str(int(box[1][1]))+','
                #                       +str(int(box[2][0]))+','+str(int(box[2][1]))+','+str(int(box[3][0]))+','+str(int(box[3][1])))
                # file.write('\r\n')



def draw_and_save_b_boxes(img, b_boxes, save_img_path):
    img = img.copy()
    for points in b_boxes:
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imwrite(save_img_path, img)


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

        print(i)
        img_name = os.path.basename(i)
        img_save_path = output_path + img_name.split('.')[0] + '_detectron_.jpg'
        img = cv2.imread(i)
        start_time = time.time()

        prediction, vis_output = detection_demo.run_on_image(img)
        vis_output.save(img_save_path)

        


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

        
        b_boxes = get_bboxes(contours)

        draw_and_save_b_boxes(img, b_boxes, img_save_path.replace('_detectron_.jpg', '.png'))
        




        csv_save_path = output_path + 'res_' + img_name.split('.')[0] + '.csv'
        save_result_to_csv(csv_save_path, prediction, b_boxes)

        print("Time: {:.2f} s / img".format(time.time() - start_time))
        img_count += 1
    print("Average Time: {:.2f} s /img".format((time.time() - start_time_all) / img_count))


