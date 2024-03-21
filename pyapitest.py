from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from shutil import rmtree

model = YOLO("luggage8_16.pt")
#results = model.predict(source="/home/ivan/videos/555/1.mp4", save_txt=True, stream=True, save_frames=True, save=True, show_boxes=False)
#results = model.predict(source="/home/nkozlitin/video/aaa.mp4", save_txt=False, stream=True, save_frames=False, save=False, show_boxes=False, conf=0.5)
#results = model.predict(source="/home/nkozlitin/cvdatastore/VideoArchive/MainData/LeftObjects/certification/AO_LO_4_SIDE.avi", save_txt=False, stream=True, save_frames=False, save=False, show_boxes=False, conf=0.25)
results = model.track(source="/home/orwell/cvdatastore/VideoArchive/MainData/LeftObjects/certification/AO_LO_4_SIDE.avi", stream=True, show=True, persist=True)

#for index,result in enumerate(results):
#    print(result)
#
#exit

images_path = Path("xxx/images")
labels_path = Path("xxx/labels")

rmtree(images_path, ignore_errors=True)
rmtree(labels_path, ignore_errors=True)

images_path.mkdir(parents=True, exist_ok=True)
labels_path.mkdir(parents=True, exist_ok=True)

with open("xxx/all.txt","w") as f:
    for index,result in enumerate(results):
        result.save_txt(f'xxx/labels/{index}.txt')
        im = Image.fromarray(result.orig_img[:,:,::-1], mode="RGB")
        im.save(f'xxx/images/{index}.jpg')
        f.write(f'./images/{index}.jpg\n')

