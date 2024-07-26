from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from shutil import rmtree
import multiprocessing
import yaml
import json
import os
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_list_yaml', nargs='?', help='list of video files')
    parser.add_argument('root_dataset_folder', nargs='?', help='root of dataset folder')
    parser.add_argument('--video_archive_root', nargs='?', help='root of video archive')
    parser.add_argument('--image_width', nargs='?', help='name of image_processor build', type=int)
    parser.add_argument('--workers_number', nargs='?', help='maximum number of profiles', type=int)
    parser.add_argument('--yolo_nn_name', nargs='?', help='profile path for autotuning')
    parser.add_argument('--settings', nargs='?', help='settings json file')
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

    with open(video_list_yaml) as video_list_file:
        supervisor_scenario = yaml.safe_load(video_list_file)

    video_list = supervisor_scenario['videofiles']
    
    images_path = root_dataset_folder / "images"
    labels_path = root_dataset_folder / "labels"
    
    rmtree(images_path, ignore_errors=True)
    rmtree(labels_path, ignore_errors=True)

    print(video_list)

    with multiprocessing.Pool(workers_number) as pool:
        videos = pool.map(process_video, video_list)
        with open(root_dataset_folder / "videos.txt", "w") as f:
            for video in videos:
                f.write(video+'\n')
