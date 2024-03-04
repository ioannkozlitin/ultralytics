from ultralytics import YOLO
from PIL import Image

model = YOLO("drone8_14.pt")
#results = model.predict(source="/home/ivan/videos/555/1.mp4", save_txt=True, stream=True, save_frames=True, save=True, show_boxes=False)
results = model.predict(source="/home/ivan/videos/555/1.mp4", save_txt=False, stream=True, save_frames=False, save=False, show_boxes=False)

for index,result in enumerate(results):
    result.save_txt(f'xxx/{index}.txt')
