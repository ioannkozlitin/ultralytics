import numpy as np
import xml.etree.ElementTree as ET
import os
import pandas as pd
from collections import defaultdict


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def get_iou(ground_truth, pred):
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
    area_of_intersection = i_height * i_width
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou

def calculate_metrics_per_frame(box_list_ideal, box_list_nn, iou_threshold=0.5):

    gt_boxes = defaultdict(list)
    pred_boxes = defaultdict(list)

    for box in box_list_ideal:
        gt_boxes[box['frame']].append(box)

    for box in box_list_nn:
        pred_boxes[box['frame']].append(box)

    TP, FP, FN, TN =0, 0, 0, 0
    total_iou = 0
    total_tp = 0

    all_frames = set(gt_boxes.keys()).union(set(pred_boxes.keys()))

    for frame in all_frames:
        gts = gt_boxes.get(frame, [])
        preds = pred_boxes.get(frame, [])
        matched_gt = set()
        matched_pred = set()

        for i, pred_box in enumerate(preds):
            best_iou = 0
            best_gt_index = None
            for j, gt_box in enumerate(gts):
                if j in matched_gt:
                    continue

                ground_truth_bbox = np.array([gt_box.get('xtl'), gt_box.get('ytl'), gt_box.get('xbr'), gt_box.get('ybr')], dtype=np.float32)
                prediction_bbox = np.array([pred_box.get('xtl'), pred_box.get('ytl'), pred_box.get('xbr'), pred_box.get('ybr')], dtype=np.float32)
                iou = get_iou(ground_truth_bbox, prediction_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = j
            if best_iou >= iou_threshold:
                TP += 1
                total_iou += best_iou
                total_tp += 1
                matched_gt.add(best_gt_index)
                matched_pred.add(i)
            #else:
            #    FP += 1

        FN += len(gts) - len(matched_gt)
        FP += len(preds) - len(matched_pred)


    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = TP /(TP + FN + FP)  if (TP + FN + FP) > 0 else 0
    mean_iou = total_iou/total_tp if total_tp > 0 else 0
    return precision, recall, accuracy, mean_iou, FP+TP, TP+FN

def extract_boxes_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    boxes = []

    for track in root.findall('track'):
        track_id = track.get('id')

        for box in track.findall('box'):
            box_info = {
                'frame': box.get('frame'),
                'xtl': box.get('xtl'),
                'ytl': box.get('ytl'),
                'xbr': box.get('xbr'),
                'ybr': box.get('ybr')
            }
            boxes.append(box_info)

    return boxes


data = {'K': [], 'Mean IOU': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'FP_TP': [], 'TP_FN': []}

name = '2024-09-13_video_2024_09_13_08_52_17_visual_narrow_mp4'
directory = '/home/neuron-2/222/2024-09-13_video_2024_09_13_08_52_17_visual_narrow_mp4/'

file_path_nn = directory + 'auto_annotation.xml'
file_path_ideal = directory + '2024-09-13_video_2024_09_13_08_52_17_visual_narrow_mp4.xml'

box_list_ideal = extract_boxes_from_xml(file_path_ideal)
box_list_nn = extract_boxes_from_xml(file_path_nn)

coef_start = 0.1

print(name)
print('Идеальные кадры:', len(box_list_ideal))
print('Кадры на выходе нейросети:', len(box_list_nn))
print('Процент несоответствия кадров:', abs(1 - len(box_list_ideal)/len(box_list_nn)))

while coef_start < 0.7:
    precision, recall, accuracy, mean_iou, FP_TP, TP_FN = calculate_metrics_per_frame(box_list_nn, box_list_ideal, coef_start)
    data['K'].append(coef_start)
    data['Mean IOU'].append(mean_iou)
    data['Accuracy'].append(accuracy)
    data['Precision'].append(precision)
    data['Recall'].append(recall)
    data['FP_TP'].append(FP_TP)
    data['TP_FN'].append(TP_FN)

    df = pd.DataFrame(data=data)
    coef_start += 0.1

print(df)

