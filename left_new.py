from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from shutil import rmtree
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xywhn2xyxy
import torchvision.transforms as T
import cv2
import numpy
import torch
from nntrain import SmokeCnnModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model2 = SmokeCnnModel(frame_number=2)
cp = torch.load("leftmodels/left1-v2.ckpt", map_location=device)

model2.load_state_dict(cp['state_dict'])
model2 = model2.to(device)
model2.eval()

model = YOLO("luggage8_16.pt")
results = model.track(source="/home/ivan/videos/VID_20230502_192209.mp4", stream=True, show=False, persist=True, conf=0.1)

stack_size = 150
image_stack=[]
new_size=(128,128)

for index,result in enumerate(results):
    image = result.orig_img
    image_stack.append(image.copy())
    if(len(image_stack) > stack_size):
        image_stack.pop(0)

    if result.boxes.is_track:
        #print(result.boxes.id)
        annotator = Annotator(image, line_width=3)
        for cls, xyxy, id in zip(result.boxes.cls, result.boxes.xyxy, result.boxes.id):
            if cls !=0:
                top_image_resized = cv2.resize(image_stack[-1][int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])], new_size)
                bottom_image_resized = cv2.resize(image_stack[0][int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])], new_size)
                image_cake = torch.stack([T.ToTensor()(numpy.concatenate([top_image_resized, bottom_image_resized],axis=2))]).to(device)
                pred = model2.net(image_cake)
                prob = torch.argmax(pred, dim=1)
                annotator.box_label(xyxy, f"{int(cls)}_{int(id)}", color=colors(3*(prob[0] > 0), True))
                #image2 = cv2.hconcat([top_image_resized, bottom_image_resized])
                #cv2.imshow("image2", image2)

    cv2.imshow("image",image)
    if cv2.waitKey(1)==27:
        break
