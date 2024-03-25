from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from shutil import rmtree

#results = model.predict(source="/home/ivan/videos/555/1.mp4", save_txt=True, stream=True, save_frames=True, save=True, show_boxes=False)
#results = model.predict(source="/home/nkozlitin/video/aaa.mp4", save_txt=False, stream=True, save_frames=False, save=False, show_boxes=False, conf=0.5)
#results = model.predict(source="/home/nkozlitin/cvdatastore/VideoArchive/MainData/LeftObjects/certification/AO_LO_4_SIDE.avi", save_txt=False, stream=True, save_frames=False, save=False, show_boxes=False, conf=0.25)
root_folder = Path("/home/nkozlitin/cvdatastore/VideoArchive/MainData")
video_list=["LeftObjects/certification/AO_LO_4_SIDE.avi","LeftObjects/certification/AO_LO_3_SIDE.avi","LeftObjects/certification/AO_LO_2_SIDE.avi"]

images_path = Path("xxx/images")
labels_path = Path("xxx/labels")

rmtree(images_path, ignore_errors=True)
rmtree(labels_path, ignore_errors=True)

images_path.mkdir(parents=True, exist_ok=True)
labels_path.mkdir(parents=True, exist_ok=True)

width = 640
with open("xxx/all.txt","w") as f:
    for video_item in video_list:
        model = YOLO("luggage8_16.pt")
        results = model.track(source=root_folder / video_item, stream=True, show=True, persist=True)
        for index,result in enumerate(results):
            image_path=Path(result.path.replace(str(root_folder), str(images_path)))
            label_path=Path(result.path.replace(str(root_folder), str(labels_path)))
            image_path.mkdir(parents=True, exist_ok=True)
            label_path.mkdir(parents=True, exist_ok=True)
            result.save_txt(str(label_path / f'{index}.txt'))
            im = Image.fromarray(result.orig_img[:,:,::-1], mode="RGB")
            height = width * im.size[1] // im.size[0]
            im_resized = im.resize((width, height))
            im_resized.save(str(image_path / f'{index}.jpg'))
            f.write(str(image_path / f'{index}.jpg')+'\n')
            if index > 100:
                break


