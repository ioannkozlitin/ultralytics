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

if __name__ == '__main__':
    root_path = Path("xxx")
    with open(root_path / "all.txt","rt") as f:
        lines = f.readlines()
    
    paths = [str(root_path.joinpath(Path(line[:-1]))) for line in lines]
    txts = img2label_paths(paths, "labels") 

    lb_stack=[]
    lb_stack_size = 150
    image_stack=[]
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
        if(len(lb_stack) > lb_stack_size):
            lb_stack.pop(0)
            image_stack.pop(0)
        #
                    
        xyxy_top = torch.tensor(xywhn2xyxy(lb[:, 1:5], w=image.shape[1], h=image.shape[0]))
        xyxy_bottom = torch.tensor(xywhn2xyxy(lb_stack[0][:, 1:5], w=image.shape[1], h=image.shape[0]))
        iou_matrix = box_iou(xyxy_top, xyxy_bottom)
        max_ious, _ = torch.max(iou_matrix,axis=1)
        print(max_ious)
                
        image_bottom = image_stack[0]
        diff_image  = ((image.astype(np.float32) - image_bottom.astype(np.float32)) / 2 + 128).astype(np.uint8)
        annotator = Annotator(diff_image, line_width=3)
        for cls, xyxy_item, max_iou in zip(lb[:,0],xyxy_top, max_ious):
            annotator.box_label(xyxy_item, f"{int(cls)}", color=colors((max_iou > 0.3) * 3, True))

        cv2.imshow("image",diff_image)
        print(image.shape)
        print(f'labels_top: {xyxy_top}')
        print(f'lables_bottom: {xyxy_bottom}')
        print(f'image path: {path_}')
        print(f'image annotation: {txt_}')

        if cv2.waitKey(1)==27:
            break


