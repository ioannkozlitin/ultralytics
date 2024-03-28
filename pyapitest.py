from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from shutil import rmtree
import multiprocessing

def process_video(video_item):
    txt_file_name = video_item.replace("/","_").replace(".","_") + ".txt"
    with open(root_dataset_folder / txt_file_name, "w") as f:
        model = YOLO(yolo_nn_name)
        results = model.track(source=root_folder / video_item, stream=True, show=False, persist=True)
        for index,result in enumerate(results):
            image_path=Path(result.path.replace(str(root_folder), str(images_path)))
            label_path=Path(result.path.replace(str(root_folder), str(labels_path)))
            image_path.mkdir(parents=True, exist_ok=True)
            label_path.mkdir(parents=True, exist_ok=True)
            result.save_txt(str(label_path / f'{index}.txt'))
            im = Image.fromarray(result.orig_img[:,:,::-1], mode="RGB")
            image_height = image_width * im.size[1] // im.size[0]
            im_resized = im.resize((image_width, image_height))
            im_resized.save(str(image_path / f'{index}.jpg'))
            f.write(str(image_path / f'{index}.jpg')+'\n')
            if index > 100:
                break
    
    return str(root_dataset_folder / txt_file_name)

if __name__ == "__main__":
    root_folder = Path("/home/nkozlitin/cvdatastore/VideoArchive/MainData")
    root_dataset_folder = Path("xxx")
    video_list=["LeftObjects/certification/AO_LO_4_SIDE.avi","LeftObjects/certification/AO_LO_3_SIDE.avi","LeftObjects/certification/AO_LO_2_SIDE.avi"]
    image_width = 640
    yolo_nn_name = "luggage8_16.pt"
    
    images_path = root_dataset_folder / "images"
    labels_path = root_dataset_folder / "labels"
    
    rmtree(images_path, ignore_errors=True)
    rmtree(labels_path, ignore_errors=True)

    with multiprocessing.Pool(None) as pool:
        videos = pool.map(process_video, video_list)
        print(videos)


