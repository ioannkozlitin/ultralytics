import cv2
import numpy as np
from annotation import Annotation
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xyxy2xywhn
from pathlib import Path
from shutil import rmtree
import random
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_name', nargs=1, help='video file name')
    parser.add_argument('root_dataset_folder', nargs=1, help='root of dataset folder')
    parser.add_argument('--video_name', help='video file name', default='')
    parser.add_argument('--nfold', nargs='?', help='nfold for train/validation split', default=5, type=int)
    parser.add_argument('--k', nargs='?', help='k for train/validation split', default=2, type=int)
    parser.add_argument('--show', help='show video', action='store_true')
    opt = parser.parse_args()

    root_dataset_folder = Path(os.path.expanduser(opt.root_dataset_folder[0]))
    annotation_name = Path(os.path.expanduser(opt.annotation_name[0]))
    video_name = str(Path(os.path.expanduser(opt.video_name)))

    images_path = root_dataset_folder / "images"
    labels_path = root_dataset_folder / "labels"

    rmtree(images_path, ignore_errors=True)
    rmtree(labels_path, ignore_errors=True)
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    annotation = Annotation()
    annotation.load(annotation_name)

    if video_name == ".":
        video_name = annotation.videofilename

    cap = cv2.VideoCapture(video_name)
        
    images_list = []
    delay = 1
    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_annotation = annotation.get_frame_poses(frame_number)

        ret, frame = cap.read()
        if not ret:
            break

        framejpg = f'{images_path}/{frame_number}.jpg'
        cv2.imwrite(framejpg, frame)
        images_list.append(f"./images/{frame_number}.jpg")

        annotator = Annotator(frame, line_width=3)
        frame_annotations=list()
        for pos in current_annotation:
            if pos.is_outside:
                continue
            xywhn = xyxy2xywhn(np.array([pos.x1, pos.y1, pos.x2, pos.y2]), w = annotation.original_width, h = annotation.original_height)
            if len(annotation.labels):
                track_label = annotation.labels[pos.track.label]
            else:
                track_label = pos.track.label
            lxywhn = xywhn.tolist()
            lxywhn.insert(0,track_label)
            frame_annotations.append(lxywhn)
            annotator.box_label((pos.x1, pos.y1, pos.x2, pos.y2), f"{track_label}_{pos.track.id}", color=colors(0, True))

        if len(current_annotation) > 0:
            anntxt = f'{labels_path}/{frame_number}.txt'
            with open(anntxt,"wt") as f:
                for ann_item in frame_annotations:
                    for el in ann_item:
                        print(el, file=f, end=' ')
                    print(file=f)

        if opt.show:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(delay)
            if key == 32:
                delay = 1 - delay
            elif key == 27:
                break
        
    nfold = 5
    k = 2
    ftr = open( root_dataset_folder / "train.txt", 'w' )
    fte = open( root_dataset_folder / "validation.txt", 'w' )
    for image_name in images_list:
        if random.randint( 1 , nfold ) == k:
            print(image_name, file=fte)
        else:
            print(image_name, file=ftr)
    
    ftr.close()
    fte.close()

    cap.release()
    cv2.destroyAllWindows()
