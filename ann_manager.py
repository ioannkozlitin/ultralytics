from tkinter import *
from tkinter import ttk
import argparse
import os
from pathlib import Path

def update_lists():
    processed_list = [str(item) for item in processed_subfolder.glob("**/*.xml")]

    videolist_names = [str(Path(item).name) for item in videolist]
    processed_list_names = [str(Path(item).name) for item in processed_list]

    videolist_names.sort()
    processed_list_names.sort()

    for item in videolist_names:
        if item not in processed_list_names:
            videolist_listbox.insert(END, item)

    for item in processed_list_names:
        processed_listbox.insert(END, item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_folder', nargs=1, help='annotation folder name')
    parser.add_argument('processed_subfolder', nargs=1, help='processed annotation subfolder name')
    opt = parser.parse_args()

    root = Tk()
    root.geometry("830x300")
    videolist_listbox = Listbox(width = 50)
    processed_listbox = Listbox(width = 50)
    videolist_listbox.grid(row=1, column=0, sticky=EW, padx=5, pady=5)
    processed_listbox.grid(row=1, column=1, sticky=EW, padx=5, pady=5)
    
    annotation_folder = Path(os.path.expanduser(opt.annotation_folder[0]))
    with open(annotation_folder / "videos.txt") as f:
        videolist = f.readlines()
    videolist = [item.strip() for item in videolist]

    processed_subfolder = annotation_folder / opt.processed_subfolder[0]
    update_lists()

    ttk.Button(text="Quit", command=root.destroy).grid(row=0, column=0, padx=5, pady=5)
    root.mainloop()
