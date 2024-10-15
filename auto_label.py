from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from shutil import rmtree
import multiprocessing
import yaml
import json
import os
import argparse
import xml.etree.ElementTree as ET
import torch
from ultralytics.utils.metrics import box_iou
import numpy as np

def process_video(video_item):
    txt_file_name = video_item.format(VideoArchive="")[1:].replace("/","_").replace(".","_") + ".txt"
    with open(root_dataset_folder / txt_file_name, "w") as f:
        model = YOLO(yolo_nn_name)
        print('filename: ' + video_item.format(VideoArchive=str(video_archive_root)))
        results = model.track(source=video_item.format(VideoArchive=str(video_archive_root)), stream=True, show=False, persist=True)
        for index,result in enumerate(results):
            image_path=Path(result.path.replace(str(video_archive_root), str(images_path)))
            label_path=Path(result.path.replace(str(video_archive_root), str(labels_path)))
            image_path.mkdir(parents=True, exist_ok=True)
            label_path.mkdir(parents=True, exist_ok=True)
            result.save_txt(str(label_path / f'{index}.txt'))
            im = Image.fromarray(result.orig_img[:,:,::-1], mode="RGB")
            image_height = image_width * im.size[1] // im.size[0]
            im_resized = im.resize((image_width, image_height))
            im_resized.save(str(image_path / f'{index}.jpg'))
            f.write(str(image_path / f'{index}.jpg')+'\n')
    
    return str(root_dataset_folder / txt_file_name)

def tracks_iou(tracks, box):
    track_boxes = np.array([[float(index)] + track["boxes"][-1][1:] for index,track in tracks.items() if len(track["boxes"])])
    if len(track_boxes):
        iou = box_iou(torch.tensor(track_boxes[:,1:]), torch.tensor([box])).numpy().reshape(-1)
        #print(f'iou: {np.argmax(iou)} {track_boxes}')
        return ((int(track_boxes[np.argmax(iou),0]),np.max(iou)))
    else:
        return (-1,0.0)

    #print(torch.tensor(track_boxes).shape, torch.tensor([box]).shape)

def process_video_xml(video_item):
    track_counter = 1
    tracks = dict()
    label_names = {4 : "plane"}
    xml_file_name = video_item.format(VideoArchive="")[1:].replace("/","_").replace(".","_") + ".xml"
    model = YOLO(yolo_nn_name)
    print('filename: ' + video_item.format(VideoArchive=str(video_archive_root)))
    results = model.track(source=video_item.format(VideoArchive=str(video_archive_root)), stream=True, show=False, persist=True, conf=0.01, iou=0.05)
    for index,result in enumerate(results):
        if index == 0:
            height, width = result.orig_shape
        boxes_data = result.boxes.data
        track_boxes = [[float(id)] + track["boxes"][-1][1:] for id, track in tracks.items() if len(track["boxes"])]
        if len(track_boxes):
            print(np.array(box_iou(result.boxes.xyxy.to(device="cpu"), torch.tensor(track_boxes)[:,1:])))
        for boxes_data_item in boxes_data:             
            box = boxes_data_item[0:4].tolist()
            box.insert(0, index)
            #
            track_id, max_iou = tracks_iou(tracks, box[1:])
            if max_iou == 0:
                tracks[track_counter] = {"boxes": list(), "label" : label_names[int(boxes_data_item[-1])], "last_index" : -1}
                track_id = track_counter
                track_counter += 1
                                
            if tracks[track_id]["last_index"] != index:
                tracks[track_id]["boxes"].append(box)
                tracks[track_id]["last_index"] = index


        if index > 100:
            break
    
    root = ET.fromstring("<annotations></annotations>\n")
    tree = ET.ElementTree(element=root)
    root.append(ET.fromstring(f"<meta><task><original_size><width>{width}</width><height>{height}</height></original_size></task></meta>"))
    for id, track in tracks.items():
        track_node = ET.Element("track", attrib={"id": str(id), "label": str(track["label"]), "source": "semi-auto"}) 
        last_index = len(track["boxes"])
        for index, box in enumerate(track["boxes"]):
            track_node.append(ET.Element("box", attrib={"frame" : str(int(box[0]))
                                                        ,"keyframe" : "1"
                                                        ,"outside"  : str(int(index == last_index-1))
                                                        ,"occluded" : "0"
                                                        ,"xtl" : str(box[1])
                                                        ,"ytl" : str(box[2])
                                                        ,"xbr" : str(box[3])
                                                        ,"ybr" : str(box[4])
                                                        ,"z_order" : "0"}))
        root.append(track_node)
    tree.write(root_dataset_folder / xml_file_name)
    return str(root_dataset_folder / xml_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_list_yaml', nargs='?', help='list of video files')
    parser.add_argument('root_dataset_folder', nargs='?', help='root of dataset folder')
    parser.add_argument('--video_archive_root', nargs='?', help='root of video archive')
    parser.add_argument('--image_width', nargs='?', help='name of image_processor build', type=int)
    parser.add_argument('--workers_number', nargs='?', help='maximum number of profiles', type=int)
    parser.add_argument('--yolo_nn_name', nargs='?', help='profile path for autotuning')
    parser.add_argument('--settings', nargs='?', help='settings json file')
    parser.add_argument('--xml_output', help='show video', action='store_true')
    opt = parser.parse_args()

    if opt.__dict__['settings'] is not None:
        with open(opt.__dict__['settings']) as settings_file:
            settings = json.load(settings_file)
    else:
        settings = dict()

    for key, value in opt.__dict__.items():
        if value is not None:
            settings[key] = value

    print(settings)

    video_archive_root = Path(os.path.expanduser(settings['video_archive_root']))
    root_dataset_folder = Path(os.path.expanduser(settings['root_dataset_folder']))
    video_list_yaml = os.path.expanduser(settings['video_list_yaml'])
    image_width = settings['image_width']
    workers_number = settings['workers_number']
    yolo_nn_name = settings['yolo_nn_name']
    xml_output = settings['xml_output']

    with open(video_list_yaml) as video_list_file:
        supervisor_scenario = yaml.safe_load(video_list_file)

    video_list = supervisor_scenario['videofiles']
    
    images_path = root_dataset_folder / "images"
    labels_path = root_dataset_folder / "labels"
    
    rmtree(images_path, ignore_errors=True)
    rmtree(labels_path, ignore_errors=True)

    print(video_list)

    with multiprocessing.Pool(workers_number) as pool:
        if xml_output:
            videos = pool.map(process_video_xml, video_list)
        else:
            videos = pool.map(process_video, video_list)

        with open(root_dataset_folder / "videos.txt", "w") as f:
            for video in videos:
                f.write(video+'\n')
