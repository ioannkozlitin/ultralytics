import cv2
import numpy as np
from annotation import Annotation
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xyxy2xywhn
from pathlib import Path
import argparse
import os

mouseX = mouseY = 0
mouseXclk = mouseYclk = 0

def mouse_event1(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONUP:
        mouseXclk, mouseYclk = x,y
    mouseX,mouseY = x,y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_name', nargs=1, help='video file name')
    parser.add_argument('--video_name', help='video file name', default='')
    opt = parser.parse_args()

    annotation_name = Path(os.path.expanduser(opt.annotation_name[0]))
    video_name = str(Path(os.path.expanduser(opt.video_name)))

    annotation = Annotation()
    annotation.load(annotation_name)

    if video_name == ".":
        video_name = annotation.videofilename

    cap = cv2.VideoCapture(video_name)
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',mouse_event1)
        
    delay = 5
    direction = 1
    frame_number = 0
    strobsize = 128
    while True:
        if (direction < 0) and (frame_number > 0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_annotation = annotation.get_frame_poses(frame_number)

        ret, frame = cap.read()
        if not ret:
            break

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
            annotator.box_label((pos.x1, pos.y1, pos.x2, pos.y2), f"{track_label}_{pos.track.id}", color=(255,0,0))

        cv2.putText(frame, f'{frame_number} / {delay}', (10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color = (0,0,0))
        cv2.rectangle(frame, (mouseX - strobsize // 2, mouseY - strobsize // 2), (mouseX + strobsize // 2, mouseY + strobsize // 2), thickness=2, color=(0,0,255))
        cv2.imshow('frame', frame)

        key = cv2.waitKey(abs(delay * direction))
        #print(mouseX, mouseY)
        #if key > 0:
        #    print(key)
        if key == 83:
            direction = min(direction+1,1)
        elif key == 81:
            direction = max(direction-1,-1)
        elif key == 84:
            delay += 5
        elif key == 82:
            delay = max(delay-5, 5)
        elif key == 85:
            strobsize *= 2
        elif key == 86:
            strobsize = max(strobsize / 2, 16)
        elif key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
