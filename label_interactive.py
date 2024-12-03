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
        self.reverse = False
        self.step = False
        self.frame_number = 0
        #
        self.startpos_button = ttk.Button(text="StartPos", command=self.fstartpos)
        self.play_button = ttk.Button(text="Play", command=self.fplay)
        self.reverse_button = ttk.Button(text="Back", command=self.freverse)
        self.step_button = ttk.Button(text="Step", command=self.fstep)
        self.frame_label = ttk.Label(text="0")
        #
        self.startpos_button.place(x = width + 5, y = 5)
        self.play_button.place(x = width + 5, y = 40)
        self.reverse_button.place(x = width + 5, y = 75)
        self.step_button.place(x = width + 5, y = 110)
        self.frame_label.place(x = width + 5, y = height - 30)
        #
        self.loop()
        self.canvas.pack()
        self.root.mainloop()

    def __del__(self):
        self.cap.release()
        
    def next_frame(self):
        if self.frame_number != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)

        ret, frame = self.cap.read()
        if ret:
            self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            self.image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.create_image(1, 1, anchor=NW, image=self.image)
        return ret
    
    def loop(self):
        if self.play:
            self.play = self.next_frame()
            if self.reverse:
                self.frame_number = max(self.frame_number - 2, 0)
            if self.step:
                self.play = False
                self.step = False
        self.frame_label.config(text = str(self.frame_number))
        self.play_button.config(text="Stop" if self.play else "Play")
        self.reverse_button.config(text="Forward" if self.reverse else "Back")
        self.root.after(self.delay, self.loop)

    def fplay(self):
        self.play = not self.play

    def freverse(self):
        self.reverse = not self.reverse
    
    def fstep(self):
        self.step = True
        self.play = True

    def fstartpos(self):
        self.frame_number = 0
        self.freverse = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_name', nargs=1, help='video file name')
    opt = parser.parse_args()

    video_name = str(Path(os.path.expanduser(opt.video_name[0])))
    Core(video_name)
