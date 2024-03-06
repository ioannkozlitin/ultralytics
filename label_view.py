from pathlib import Path
from ultralytics.data import Annotator, colors
from ultralytics.utils.ops import xywhn2xyxy
import numpy as np
import os
import cv2

def img2label_paths(img_paths, label_folder):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}{label_folder}{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

if __name__ == '__main__':
    root_path = Path("/home/orwell/proj/ssd0/ikozlitin/ADE20K_2021_17_01")
    with open(root_path / "validation.txt","rt") as f:
        lines = f.readlines()
    
    paths = [str(root_path.joinpath(Path(line[:-1]))) for line in lines]
    txts = img2label_paths(paths, "luggage_24") 

    for path_,txt_ in zip(paths, txts):
        image = cv2.imread(str(path_))
        annotator = Annotator(image, line_width=3)
        
        with open(txt_) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
            xyxy = xywhn2xyxy(lb[:, 1:5], w=image.shape[1], h=image.shape[0])
            for xyxy_item in xyxy:
                annotator.box_label(xyxy_item, "object", color=colors(lb[0][0], True))

        cv2.imshow("image",image)
        print(image.shape)
        print(f'labels: {xyxy}')
        print(f'image path: {path_}')
        print(f'image annotation: {txt_}')

        if cv2.waitKey(0)==27:
            break


