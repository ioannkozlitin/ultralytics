import cv2
import numpy as np
from annotation import Annotation
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xyxy2xywhn
from pathlib import Path
from shutil import rmtree

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
    print(f'{len(annotation.frame_poses)}')

    cap = cv2.VideoCapture(videoname)

    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_annotation = annotation.get_frame_poses(frame_number)

        ret, frame = cap.read()
        if not ret:
            break

        annotator = Annotator(frame, line_width=3)
        frame_annotations=list()
        for pos in current_annotation:
            xywhn = xyxy2xywhn(np.array([pos.x1, pos.y1, pos.x2, pos.y2]), w = annotation.original_width, h = annotation.original_height)
            track_label = annotation.labels[pos.track.label]
            lxywhn = xywhn.tolist()
            lxywhn.insert(0,track_label)
            frame_annotations.append(lxywhn)
            annotator.box_label((pos.x1, pos.y1, pos.x2, pos.y2), f"{track_label}", color=colors(0, True))

        print(frame_annotations)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
