from ultralytics import YOLO
from PIL import Image
from pathlib import Path

model = YOLO("yolov8n.pt")
#results = model.predict(source="/home/ivan/videos/555/1.mp4", save_txt=True, stream=True, save_frames=True, save=True, show_boxes=False)
results = model.predict(source="/home/nkozlitin/video/aaa.mp4", save_txt=False, stream=True, save_frames=False, save=True, show_boxes=True, conf=0.1)

Path("xxx/images").mkdir(parents=True, exist_ok=True)
Path("xxx/labels").mkdir(parents=True, exist_ok=True)

for index,result in enumerate(results):
    result.save_txt(f'xxx/labels/{index}.txt')
    im = Image.fromarray(result.orig_img[:,:,::-1], mode="RGB")
    im.save(f'xxx/images/{index}.jpg')
