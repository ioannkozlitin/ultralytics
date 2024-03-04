from ultralytics import YOLO
from PIL import Image

model = YOLO("drone8_14.pt")
results = model.predict(source="/home/nkozlitin/Videos/555/1.mp4", save_txt=True, stream=True, save_frames=True, save=True, show_boxes=False)

for result in results:
    pass
