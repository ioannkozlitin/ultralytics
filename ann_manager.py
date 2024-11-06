from tkinter import *
from tkinter import ttk
import argparse
import os
from pathlib import Path
import subprocess
from shutil import rmtree
from annotation import Annotation
import multiprocessing
import cv2
from ultralytics.utils.ops import xyxy2xywhn
import numpy as np
import yaml
import json
import random
import cv2

def update_lists():
    global videolist, processed_list
    videolist = [str(item) for item in annotation_folder.glob("**/*.xml")]
    processed_list = [str(item) for item in processed_subfolder.glob("**/*.xml")]

    videolist.sort()
    processed_list.sort()

    processed_list_names = [str(Path(item).relative_to(processed_subfolder)) for item in processed_list]
    
    videolist_listbox.delete(0,videolist_listbox.size()-1)
    for item in videolist:
        short_item = str(Path(item).relative_to(annotation_folder))
        if short_item not in processed_list_names:
            videolist_listbox.insert(END, short_item)

    processed_listbox.delete(0,processed_listbox.size()-1)
    for item in processed_list_names:
        processed_listbox.insert(END, item)

def process_video():
    selection = videolist_listbox.curselection()
    if len(selection):
        print(videolist_listbox.get(selection))
        subprocess.run(["python3", "annxml_view.py", annotation_folder / videolist_listbox.get(selection), "--annotation_folder", opt.processed_subfolder[0]])
        update_lists()

def view_video():
    selection = processed_listbox.curselection()
    if len(selection):
        subprocess.run(["python3", "annxml_view.py", processed_subfolder / processed_listbox.get(selection)])

def process_video_item(item):
    print('Process ', item)
    annotation = Annotation()
    annotation.load(item)
    cap = cv2.VideoCapture(annotation.videofilename)
    root_images_path = Path(dataset_folder_entry.get()) / "images"
    images_path = root_images_path / Path(item).stem
    rmtree(images_path, ignore_errors=True)
    images_path.mkdir(parents=True, exist_ok=True)
    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
            
        framejpg = f'{images_path}/{frame_number}.jpg'
        cv2.imwrite(framejpg, frame)
    print(item, ' processed')

def process_annotation(item):
    print('Process ', item)

    labels_path = Path(dataset_folder_entry.get()) / "labels" / Path(item).stem
    images_path = Path(dataset_folder_entry.get()) / "images" / Path(item).stem

    rmtree(labels_path, ignore_errors=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    annotation = Annotation()
    annotation.load(item)
    images = []
    for frame_number, frame_pos in enumerate(annotation.frame_poses):
        frame_annotations=list()
        for pos in frame_pos:
            if pos.is_outside:
                continue
            xywhn = xyxy2xywhn(np.array([pos.x1, pos.y1, pos.x2, pos.y2]), w = annotation.original_width, h = annotation.original_height)
            if len(annotation.labels):
                track_label = annotation.labels[pos.track.label]
            else:
                track_label = inv_label_names[pos.track.label]
            lxywhn = xywhn.tolist()
            lxywhn.insert(0,track_label)
            frame_annotations.append(lxywhn)
            
        if len(annotation.frame_poses):
            anntxt = f'{labels_path}/{frame_number}.txt'
            image_name = f'{images_path}/{frame_number}.jpg'
            images.append(image_name)
            with open(anntxt,"wt") as f:
                for ann_item in frame_annotations:
                    for el in ann_item:
                        print(el, file=f, end=' ')
                    print(file=f)

    print(item, f' processed images: {len(images)}')
    return images

def generate_images():
    with multiprocessing.Pool(workers_number) as pool:
            pool.map(process_video_item, videolist)

def generate_labels():
    with multiprocessing.Pool(workers_number) as pool:
        all_images = pool.map(process_annotation, processed_list)
    
    all_images = [item for image_item in all_images for item in image_item]

    nfold = 5
    k = 2
    ftr = open( Path(dataset_folder_entry.get()) / "train.txt", 'w' )
    fte = open( Path(dataset_folder_entry.get()) / "validation.txt", 'w' )
    for image_name in all_images:
        if random.randint( 1 , nfold ) == k:
            print(image_name, file=fte)
        else:
            print(image_name, file=ftr)
    
    ftr.close()
    fte.close()

def search_sources():
    source_folder = Path(source_folder_entry.get())
    sourcelist = [str(item.relative_to(source_folder)) for item in source_folder.glob(search_pattern_entry.get())]
    source_listbox.delete(0,source_listbox.size()-1)
    for item in sourcelist:
        source_listbox.insert(END, item)

def run_auto_label():
    source_list = selected_listbox.get(0, selected_listbox.size()-1)
    sourcelist_yaml = {"videofiles": ["{VideoArchive}/"+item for item in source_list]}
    with open("videolist.yaml","w") as f:
        yaml.dump(sourcelist_yaml, f, allow_unicode=True)

    with open("ann_manager_settings.json") as f:
        settings = json.load(f)

    settings["video_list_yaml"] = "videolist.yaml"
    settings["root_dataset_folder"] = str(annotation_folder)
    settings["video_archive_root"] = str(source_folder)

    with open("auto_label_settings_.json","w") as f:
        settings = json.dump(settings, f, indent=4, ensure_ascii=False)
    subprocess.run(["python3", "auto_label.py", "--settings", "auto_label_settings_.json", "--xml_output"])
    update_lists()

def select_video():
    curselection = source_listbox.curselection()
    if len(curselection):
        selection = source_listbox.get(curselection)
        selected_data = selected_listbox.get(0, selected_listbox.size()-1)
        if selection not in selected_data:
            selected_listbox.insert(END, selection)

def select_all_video():
    source_list = source_listbox.get(0, source_listbox.size()-1)
    selected_data = selected_listbox.get(0, selected_listbox.size()-1)
    for selection in source_list:
        if selection not in selected_data:
            selected_listbox.insert(END, selection)

def remove_item():
    curselection = selected_listbox.curselection()
    selected_listbox.delete(curselection)

def remove_all_items():
    selected_listbox.delete(0, selected_listbox.size()-1)

def video_preview():
    curselection = selected_listbox.curselection()
    video_name = selected_listbox.get(curselection)
    #print(video_name)
    cap = cv2.VideoCapture(source_folder / video_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_folder', nargs=1, help='annotation folder name')
    parser.add_argument('processed_subfolder', nargs=1, help='processed annotation subfolder name')
    parser.add_argument('--workers_number', nargs='?', help='maximum number of processes', type=int, default=8)
    parser.add_argument('--label_names_yaml', nargs='?', help='yaml file with label names', default='')
    parser.add_argument('--source_folder', nargs='?', help='source folder name', default='')
    opt = parser.parse_args()

    workers_number = opt.workers_number
    if Path(opt.label_names_yaml).is_file():
        with open(opt.label_names_yaml) as f:
            label_names = yaml.safe_load(f)
    else:
        label_names = None

    if label_names is None:
        inv_label_names = None
    else:
        inv_label_names = {value : key for key,value in label_names.items()}

    print(inv_label_names)

    root = Tk()
    root.geometry("830x600")
    root.resizable(False, False)

    videolist_listbox = Listbox(width = 50)
    processed_listbox = Listbox(width = 50)
    videolist_listbox.grid(row=1, column=0, sticky=EW, padx=5, pady=5)
    processed_listbox.grid(row=1, column=1, sticky=EW, padx=5, pady=5)
    
    annotation_folder = Path(os.path.expanduser(opt.annotation_folder[0]))
    source_folder = Path(os.path.expanduser(opt.source_folder))

    processed_subfolder = annotation_folder / opt.processed_subfolder[0]

    videolist=[]
    processed_list=[]
    videolist_listbox_fullnames=[]
    update_lists()

    ttk.Button(text="Process", command=process_video).grid(row=0, column=0, padx=5, pady=5)
    ttk.Button(text="View", command=view_video).grid(row=0, column=1, padx=5, pady=5)
    ttk.Label(text = "Dataset folder").grid(row=2, column=0, padx=5, pady=5, sticky=W)

    dataset_folder_entry = ttk.Entry(width=35)
    dataset_folder_entry.insert(0, annotation_folder)
    dataset_folder_entry.grid(row=2, column=0, padx=5, pady=5, sticky=E)

    ttk.Button(text="Generate images", command=generate_images).grid(row=3, column=0, padx=5, pady=5, sticky=W)
    ttk.Button(text="Generate labels", command=generate_labels).grid(row=3, column=0, padx=130, pady=5, sticky=W)

    source_folder_entry = ttk.Entry(width=35)
    source_folder_entry.insert(0, source_folder)
    source_folder_entry.grid(row=2, column=1, padx=5, pady=5, sticky=E)
    ttk.Label(text = "Source folder").grid(row=2, column=1, padx=5, pady=5, sticky=W)

    search_pattern_entry = ttk.Entry(width=35)
    search_pattern_entry.insert(0, "**/*.mp4")
    search_pattern_entry.grid(row=3, column=1, padx=5, pady=5, sticky=E)
    ttk.Label(text = "Search pattern").grid(row=3, column=1, padx=5, pady=5, sticky=W)

    source_listbox = Listbox(width = 50)
    source_listbox.grid(row=4, column=1, sticky=EW, padx=5, pady=5)

    selected_listbox = Listbox(width = 50)
    selected_listbox.grid(row=4, column=0, sticky=EW, padx=5, pady=5)

    ttk.Button(text="Search", command=search_sources).grid(row=5, column=1, padx=5, pady=5, sticky=W)
    ttk.Button(text="Select", command=select_video).grid(row=5, column=1, padx=95, pady=5, sticky=W)
    ttk.Button(text="Select all", command=select_all_video).grid(row=5, column=1, padx=185, pady=5, sticky=W)
    ttk.Button(text="Autolabel", command=run_auto_label).grid(row=5, column=0, padx=5, pady=5, sticky=W)
    ttk.Button(text="Remove", command=remove_item).grid(row=5, column=0, padx=95, pady=5, sticky=W)
    ttk.Button(text="Remove all", command=remove_all_items).grid(row=5, column=0, padx=185, pady=5, sticky=W)
    ttk.Button(text="Preview", command=video_preview).grid(row=5, column=0, padx=5, pady=5, sticky=E)
    root.mainloop()
