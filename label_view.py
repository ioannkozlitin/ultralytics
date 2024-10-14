from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xywhn2xyxy
import numpy as np
import os
import cv2

def img2label_paths(img_paths, label_folder):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}{label_folder}{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

if __name__ == '__main__':
    root_path = Path("/home/neuron-2/datasets/ccc0")
    with open(root_path / "all.txt","rt") as f:
        lines = f.readlines()
    
    paths = [str(root_path.joinpath(Path(line[:-1]))) for line in lines]
    txts = img2label_paths(paths, "labels") 

    for path_,txt_ in zip(paths, txts):
        image = cv2.imread(str(path_))
        annotator = Annotator(image, line_width=3)

        if not os.path.exists(txt_):
            continue
        
        with open(txt_) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if len(lb):
                lb = np.array(lb, dtype=np.float32)
                xyxy = xywhn2xyxy(lb[:, 1:5], w=image.shape[1], h=image.shape[0])
                for cls, xyxy_item in zip(lb[:,0],xyxy):
                    annotator.box_label(xyxy_item, f"{int(cls)}", color=colors(cls, True))
        
        if len(lb) < 1:
            continue
        
        cv2.imshow("image",image)
        print(image.shape)
        print(f'labels: {xyxy}')
        print(f'image path: {path_}')
        print(f'image annotation: {txt_}')

        if cv2.waitKey(0)==27:
            break

