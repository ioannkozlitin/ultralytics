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

def update_lists():
    global videolist, processed_list, videolist_listbox_fullnames
    videolist = [str(item) for item in annotation_folder.glob("**/*.xml")]
    processed_list = [str(item) for item in processed_subfolder.glob("**/*.xml")]

    videolist.sort()
    processed_list.sort()

    processed_list_names = [str(Path(item).name) for item in processed_list]
    
    videolist_listbox.delete(0,videolist_listbox.size()-1)
    videolist_listbox_fullnames = []
    for item in videolist:
        short_item = str(Path(item).name)
        if short_item not in processed_list_names:
            videolist_listbox_fullnames.append(item)
            videolist_listbox.insert(END, short_item)

    processed_listbox.delete(0,processed_listbox.size()-1)
    for item in processed_list_names:
        processed_listbox.insert(END, item)

def process_video():
    selection = videolist_listbox.curselection()
    if len(selection):
        subprocess.run(["python3", "annxml_view.py", videolist_listbox_fullnames[selection[0]], "--annotation_folder", opt.processed_subfolder[0]])
        update_lists()

def view_video():
    selection = processed_listbox.curselection()
    if len(selection):
        print(processed_list[selection[0]])
        subprocess.run(["python3", "annxml_view.py", processed_list[selection[0]]])

def process_video(item):
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

def generate_images():
    with multiprocessing.Pool(workers_number) as pool:
            pool.map(process_video, videolist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_folder', nargs=1, help='annotation folder name')
    parser.add_argument('processed_subfolder', nargs=1, help='processed annotation subfolder name')
    parser.add_argument('--workers_number', nargs='?', help='maximum number of processes', type=int, default=8)
    opt = parser.parse_args()

    workers_number = opt.workers_number

    root = Tk()
    root.geometry("830x330")
    root.resizable(False, False)

    videolist_listbox = Listbox(width = 50)
    processed_listbox = Listbox(width = 50)
    videolist_listbox.grid(row=1, column=0, sticky=EW, padx=5, pady=5)
    processed_listbox.grid(row=1, column=1, sticky=EW, padx=5, pady=5)
    
    annotation_folder = Path(os.path.expanduser(opt.annotation_folder[0]))

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
    #ttk.Button(text="Generate labels", command=generate_images).grid(row=3, column=0, padx=130, pady=5, sticky=W)
    root.mainloop()
