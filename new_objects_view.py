from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.metrics import box_iou
import torch
import numpy as np
import os
import cv2

def img2label_paths(img_paths, label_folder):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}{label_folder}{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def process_video(txt_labels_filename, stack_size, foutput, display=False):
    with open(txt_labels_filename,"rt") as f:
        lines = f.readlines()
    
    paths = [line.strip() for line in lines if len(line.strip())]
    txts = img2label_paths(paths, "labels")
    
    lb_stack=[]
    image_stack=[]
    path_stack=[]

    for path_,txt_ in zip(paths, txts):
        image = cv2.imread(str(path_))

        try:
            with open(txt_) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        except:
            pass
        
        lb = [lb_item for lb_item in lb if int(lb_item[0]) != 0]
        lb = lb if len(lb) else [[24,2,2,0,0]]
                
        lb = np.array(lb, dtype=np.float32)
        #
        lb_stack.append(lb)
        image_stack.append(image.copy())
        path_stack.append(str(path_))
        if(len(lb_stack) > stack_size):
            lb_stack.pop(0)
            image_stack.pop(0)
            path_stack.pop(0)
        # 
                    
        xywh_top = lb[:, 1:5]
        xyxy_top = torch.tensor(xywhn2xyxy(lb[:, 1:5], w=image.shape[1], h=image.shape[0]))
        xyxy_bottom = torch.tensor(xywhn2xyxy(lb_stack[0][:, 1:5], w=image.shape[1], h=image.shape[0]))
        iou_matrix = box_iou(xyxy_top, xyxy_bottom)
        max_ious, _ = torch.max(iou_matrix,axis=1)
        bottom_image_name = Path(path_stack[0])
        top_image_name = Path(path_stack[-1])
        #print(max_ious)
                
        image_bottom = image_stack[0]
        diff_image  = ((image.astype(np.float32) - image_bottom.astype(np.float32)) / 2 + 128).astype(np.uint8)
        annotator = Annotator(diff_image, line_width=3)
        for cls, xyxy_item, max_iou, xywh_item in zip(lb[:,0],xyxy_top, max_ious, xywh_top):
            if xyxy_item[0] <= image.shape[1]:
                iou_condition = int(max_iou > 0.8)
                annotator.box_label(xyxy_item, f"{int(cls)}", color=colors(iou_condition * 3, True))
                line = f'{bottom_image_name} {top_image_name} '
                for value in xywh_item:
                    line += str(value)+' '
                line += f'{iou_condition} {int(cls)}\n'
                foutput.write(line)

        print(f'{top_image_name}')

        if display:
            cv2.imshow("image",diff_image)
            if cv2.waitKey(1)==27:
                break

if __name__ == '__main__':
    root_path = Path("xxx")
    with open(root_path / "videos.txt","rt") as f:
        videos = f.readlines()

    videos = [item.strip() for item in videos if len(item.strip())]
    with open(root_path / "spec_labels.txt","wt") as ff:
        for video in videos:
            process_video(video, 150, ff)
