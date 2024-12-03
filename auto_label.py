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
        results = model.track(source=video_item.format(VideoArchive=str(video_archive_root)), stream=True, show=False, persist=True, conf=conf, iou=iou)
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

class Track:
    def __init__(self, box, label, frame):
        self.boxes = [box]
        self.frames = [frame]
        self.label = label
        self.active = True

    def add(self, box, frame):
        if self.frames[-1] != frame:
            self.boxes.append(box)
            self.frames.append(frame)

    def frame_boxes(self):
        return zip(self.frames, self.boxes)

class TrackStore:
    def __init__(self, width, height):
        self.tracks=[]
        self.width = width
        self.height = height

    def add_track(self, box, label, frame):
        self.tracks.append(Track(box, label, frame))

    def add_box_to_track(self, n, box, frame):
        self.tracks[n].add(box, frame)

    def add_best_boxes_to_tracks(self, boxes_xyxy, frame):
        iou_matrix = self.track_boxes_iou_matrix(boxes_xyxy)
        if iou_matrix is None:
            return [n for n, _ in enumerate(boxes_xyxy)]
        iou_matrix_shape = iou_matrix.shape
        track_ids_ = self.track_ids()
        box_enable = [True] * iou_matrix_shape[0]
        track_enable = [True] * iou_matrix_shape[1]
        run = True
        while run:
            max_iou = 0
            best_box_id = 0
            best_track_number = 0
            run = False
            for track_number, track_enable_sign in enumerate(track_enable):
                if track_enable_sign:
                    for box_id, box_enable_sign in enumerate(box_enable):
                        if box_enable_sign:
                            if(iou_matrix[box_id, track_number] > max_iou):
                                run = True
                                max_iou = iou_matrix[box_id, track_number]
                                best_box_id = box_id
                                best_track_number = track_number
            if run:
                track_enable[best_track_number] = False
                box_enable[best_box_id] = False
                self.add_box_to_track(track_ids_[best_track_number], boxes_xyxy[best_box_id], frame)
                #print(best_box_id, track_ids_[best_track_number], max_iou)
        return [n for n, enable_flag in enumerate(box_enable) if enable_flag]

    def empty(self):
        return len(self.tracks) == 0
    
    def track_boxes(self):
        return torch.tensor(np.array([track.boxes[-1].numpy() for track in self.tracks if track.active]))
    
    def track_ids(self):
        return [id for id, track in enumerate(self.tracks) if track.active]

    def track_boxes_iou_matrix(self, boxes_xyxy):
        #print(f'TRACK BOXES {self.tracks}')
        track_boxes = self.track_boxes()
        if len(track_boxes):
            return np.array(box_iou(boxes_xyxy, self.track_boxes()))
        else:
            return None
        
    def delete_old_tracks(self, frame, wait_frames):
        for track in self.tracks:
            if len(track.frames):
                if(frame - track.frames[-1] > wait_frames):
                    track.active = False

    def dump(self, xml_file_name, video_name):
        if len(self.tracks) == 0:
            return False
        
        root = ET.fromstring("<annotations></annotations>\n")
        tree = ET.ElementTree(element=root)
        root.append(ET.fromstring(f"<meta><task><original_size><width>{self.width}</width><height>{self.height}</height></original_size><source>{video_name}</source></task></meta>"))
        for id, track in enumerate(self.tracks):
            track_node = ET.Element("track", attrib={"id": str(id), "label": str(track.label), "source": "semi-auto"}) 
            track_length = len(track.boxes)
            if track_length < min_track_length:
                continue
            for index, (frame, box) in enumerate(track.frame_boxes()):
                box = box.numpy()
                track_node.append(ET.Element("box", attrib={"frame" : str(frame)
                                                        ,"keyframe" : "1"
                                                        ,"outside"  : str(int(index == track_length-1))
                                                        ,"occluded" : "0"
                                                        ,"xtl" : str(box[0])
                                                        ,"ytl" : str(box[1])
                                                        ,"xbr" : str(box[2])
                                                        ,"ybr" : str(box[3])
                                                        ,"z_order" : "0"}))
            root.append(track_node)
        tree.write(xml_file_name)        
        return True

def process_video_xml(video_item):
    xml_file_name = video_item.format(VideoArchive="")[1:].replace("/","_").replace(".","_") + ".xml"
    video_name = video_item.format(VideoArchive=str(video_archive_root))
    model = YOLO(yolo_nn_name)
    print('filename: ' + video_item.format(VideoArchive=str(video_archive_root)))
    #results = model.track(source=video_item.format(VideoArchive=str(video_archive_root)), stream=True, show=False, persist=True, conf=conf, iou=iou)
    results = model.predict(source=video_name, stream=True, show=False, conf=conf, iou=iou, verbose=False)
    trackstore = None
    for frame_number,result in enumerate(results):
        if trackstore is None:
            if label_names is None:
                current_label_names = result.names
            else:
                current_label_names = label_names
            height, width = result.orig_shape
            trackstore = TrackStore(width, height)
        boxes_xyxy = result.boxes.xyxy.cpu()
        notrack_box_ids = trackstore.add_best_boxes_to_tracks(boxes_xyxy, frame_number)
        #print(notrack_box_ids)
        for id in notrack_box_ids:
            box_data = result.boxes.data[id]
            if box_data[4] > capture_conf:
                label_id = int(box_data[-1])
                if label_id in current_label_names:
                    trackstore.add_track(box=box_data[0:4].cpu(), label=current_label_names[label_id], frame=frame_number)
        trackstore.delete_old_tracks(frame_number, wait_frames)
        
        #if frame_number > 10:
        #    break

    print("done: ", root_dataset_folder / xml_file_name)

    root_dataset_folder.mkdir(parents=True, exist_ok=True)
    if trackstore.dump(root_dataset_folder / xml_file_name, video_name):
        return str(root_dataset_folder / xml_file_name)
    else:
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_list_yaml', nargs='?', help='list of video files')
    parser.add_argument('root_dataset_folder', nargs='?', help='root of dataset folder')
    parser.add_argument('--video_archive_root', nargs='?', help='root of video archive')
    parser.add_argument('--image_width', nargs='?', help='name of image_processor build', type=int)
    parser.add_argument('--workers_number', nargs='?', help='maximum number of profiles', type=int)
    parser.add_argument('--yolo_nn_name', nargs='?', help='yolo neural network name')
    parser.add_argument('--settings', nargs='?', help='settings json file')
    parser.add_argument('--xml_output', help='show video', action='store_true')
    parser.add_argument('--conf', nargs='?', help='confidence threshold')
    parser.add_argument('--capture_conf', nargs='?', help='capture confidence threshold')
    parser.add_argument('--iou', nargs='?', help='iou nms')
    parser.add_argument('--label_names_yaml', nargs='?', help='yaml file with label names')
    parser.add_argument('--wait_frames', nargs='?', help='number of frames before delete track')
    parser.add_argument('--min_track_length', nargs='?', help='minimal length of track')
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
    conf = float(settings['conf'])
    capture_conf = float(settings['capture_conf'])
    iou = float(settings['iou'])
    label_names_yaml = settings['label_names_yaml']
    wait_frames = settings['wait_frames']
    min_track_length = settings['min_track_length']

    with open(video_list_yaml) as video_list_file:
        supervisor_scenario = yaml.safe_load(video_list_file)

    if Path(label_names_yaml).is_file():
        with open(label_names_yaml) as f:
            label_names = yaml.safe_load(f)
    else:
        label_names = None

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
                if video != "":
                    f.write(video+'\n')
