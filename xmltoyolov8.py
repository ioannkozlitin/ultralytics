import cv2
import numpy as np
from annotation import Annotation
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xyxy2xywhn
from pathlib import Path
from shutil import rmtree
import random

if __name__ == '__main__':
    annname = "/home/neuron-2/Видео/new/DJI_2024_09_20_15_23_21/annotations.xml"
    videoname = "/home/neuron-2/Видео/new/2024_09_20_15_23_21_visual_narrow.mp4"
    root_dataset_folder = Path("/home/neuron-2/Видео/new/DJI_2024_09_20_15_23_21")

    images_path = root_dataset_folder / "images"
    labels_path = root_dataset_folder / "labels"

    rmtree(images_path, ignore_errors=True)
    rmtree(labels_path, ignore_errors=True)
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    annotation = Annotation()
    annotation.load(annname)

    cap = cv2.VideoCapture(videoname)

    images_list = []
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
            xywhn = xyxy2xywhn(np.array([pos.x1, pos.y1, pos.x2, pos.y2]), w = annotation.original_width, h = annotation.original_height)
            track_label = annotation.labels[pos.track.label]
            lxywhn = xywhn.tolist()
            lxywhn.insert(0,track_label)
            frame_annotations.append(lxywhn)
            annotator.box_label((pos.x1, pos.y1, pos.x2, pos.y2), f"{track_label}", color=colors(0, True))

        if len(current_annotation) > 0:
            anntxt = f'{labels_path}/{frame_number}.txt'
            with open(anntxt,"wt") as f:
                for ann_item in frame_annotations:
                    for el in ann_item:
                        print(el, file=f, end=' ')
                    print(file=f)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
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
