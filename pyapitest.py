from ultralytics import YOLO
from PIL import Image
from pathlib import Path

model = YOLO("drone8_14.pt")
#results = model.predict(source="/home/ivan/videos/555/1.mp4", save_txt=True, stream=True, save_frames=True, save=True, show_boxes=False)
results = model.predict(source="ultralytics/assets/bus.jpg", save_txt=False, stream=True, save_frames=False, save=False, show_boxes=False)

Path("xxx/images").mkdir(parents=True, exist_ok=True)
Path("xxx/labels").mkdir(parents=True, exist_ok=True)

for index,result in enumerate(results):
    result.save_txt(f'xxx/labels/{index}.txt')
    im = Image.fromarray(result.orig_img[:,:,::-1], mode="RGB")
    im.save(f'xxx/images/{index}.jpg')
