import cv2
import numpy as np
from annotation import Annotation
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xyxy2xywhn
from pathlib import Path
import argparse
import os

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

class Core:
    def __init__(self, video_name):
        self.root = Tk()
        self.cap = cv2.VideoCapture(video_name)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = Canvas(self.root, width=width+100, height=height, bg='white')
        self.delay = 5
        self.play = False
        self.loop()
        #
        ttk.Button(text="Start", command=self.start).place(x = width + 5, y = 5)
        ttk.Button(text="Play", command=self.fplay).place(x = width + 5, y = 35)
        ttk.Button(text="Stop", command=self.fstop).place(x = width + 5, y = 65)
        #
        self.canvas.pack()
        self.root.mainloop()

    def __del__(self):
        self.cap.release()
        
    def next_frame(self):
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            self.image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.create_image(1, 1, anchor=NW, image=self.image)
        return ret
    
    def loop(self):
        if self.play:
            self.play = self.next_frame()
        self.root.after(self.delay, self.loop)

    def start(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.fplay()

    def fplay(self):
        self.play = True


    def fstop(self):
        self.play = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_name', nargs=1, help='video file name')
    opt = parser.parse_args()

    video_name = str(Path(os.path.expanduser(opt.video_name[0])))
    Core(video_name)
