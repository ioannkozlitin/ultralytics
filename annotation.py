from collections import defaultdict
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import cv2


class Track:
    def __init__(self, global_frame_poses_list, id, label, is_auto):
        self.global_frame_poses_list = global_frame_poses_list
        self.id = id
        self.label = label
        self.is_auto = is_auto
        self.frames = []
        self.matched_ids = set()  #  id всех парных треков при сопоставлении 1:1
        self.all_match_ids = set()  #  id всех парных треков триплет-статистики (для FAR,FRR)

    def clear(self):
        ''' break cyclic links to allow object to be destroyed'''
        self.frames.clear()
        self.global_frame_poses_list = None

    def add_pos(self, pos):
        if self.frames:
            assert pos.frame_no > self.frames[-1].frame_no
            if pos.frame_no > self.frames[-1].frame_no + 1 and not self.frames[-1].is_outside:
                self.interpolate(self.frames[-1], pos)
        self._add_pos(pos)

    def interpolate(self, pos1, pos2):
        for n in range(pos1.frame_no + 1, pos2.frame_no):
            pos = pos1.copy()
            pos.frame_no = n
            pos.is_keyframe = False
            pos.x1 = self._interpolate_val(pos1.x1, pos2.x1, pos1.frame_no, pos2.frame_no, n)
            pos.y1 = self._interpolate_val(pos1.y1, pos2.y1, pos1.frame_no, pos2.frame_no, n)
            pos.x2 = self._interpolate_val(pos1.x2, pos2.x2, pos1.frame_no, pos2.frame_no, n)
            pos.y2 = self._interpolate_val(pos1.y2, pos2.y2, pos1.frame_no, pos2.frame_no, n)
            self._add_pos(pos)

    def _interpolate_val(self, v1, v2, f1, f2, f):
        return v1 + (v2-v1)*(f-f1)/(f2-f1)

    def _add_pos(self, pos):
        self.frames.append(pos)
        while pos.frame_no >= len(self.global_frame_poses_list):
            self.global_frame_poses_list.append(set())
        self.global_frame_poses_list[pos.frame_no].add(pos)

    def revise_keyframe_flags(self, add_key=False, remove_key=False, pos_tolerance = None):
        """
        Ставит is_keyframe=False для кадров, на которых рамка имеет те же атрибуты, что и на предыдущем,
        и по положению отличается не более чем на pos_thr пикселей
        """
        def is_same(prev_pos, pos):
            same_position = True
            if pos_tolerance is not None:
                same_position = (abs(pos.x1 - prev_pos.x1) <= pos_tolerance and
                                abs(pos.x2 - prev_pos.x2) <= pos_tolerance and
                                abs(pos.y1 - prev_pos.y1) <= pos_tolerance and
                                abs(pos.y2 - prev_pos.y2) <= pos_tolerance)
            return (same_position and
                    pos.is_outside == prev_pos.is_outside and
                    pos.is_occluded == prev_pos.is_occluded and
                    pos.is_occluded == prev_pos.is_occluded and
                    pos.matched_id == prev_pos.matched_id and
                    str(sorted(pos.all_match_ids)) == str(sorted(prev_pos.all_match_ids)) and
                    str(pos.attributes) == str(prev_pos.attributes)
                    )
        prev_pos = None
        for pos in sorted(self.frames, key=lambda p: p.frame_no):
            if prev_pos is not None and is_same(prev_pos, pos):
                if remove_key:
                    pos.is_keyframe = False
            else:
                if add_key:
                    pos.is_keyframe = True
            if pos.is_keyframe:
                prev_pos = pos


class Pos:
    def __init__(self, track, frame_no, is_outside, is_occluded, is_keyframe, x1, y1, x2, y2):
        self.track = track
        self.frame_no = frame_no
        self.is_outside = is_outside
        self.is_occluded = is_occluded
        self.is_keyframe = is_keyframe
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.matched_id = None   #  id парного трека при сопоставлении 1:1
        self.all_match_ids = set()  #  id всех парных треков триплет-статистики (для FAR,FRR)
        self.attributes = defaultdict(list)  # list of given attribute values

    def copy(self):
        new_pos = copy.copy(self)
        new_pos.all_match_ids = copy.copy(self.all_match_ids)
        new_pos.attributes = copy.copy(self.attributes)
        return new_pos

class Annotation:
    def __init__(self):
        self.filename = None
        self.tracks = dict()  # track id : track
        self.frame_poses = list()  # frame_no -> set(Pos)
        self.labels = dict()
        self.original_width = 0
        self.original_height = 0
        self.idealfile = None

    def __del__(self):
        self.clear()

    def get_frame_poses(self, frame_no):
        if frame_no >= 0 and frame_no < len(self.frame_poses):
            return self.frame_poses[frame_no]
        else:
            return set()


    def clear(self):
        ''' break cyclic links to allow object to be destroyed'''
        for id, track in self.tracks.items():
            track.clear()
        self.tracks.clear()
        self.frame_poses.clear()

    def load(self, filename):
        assert Path(filename).is_file(), "annotation file not exists: " + filename
        self.clear()
        self.filename = filename
        if Path(filename).suffix == '.zip':
            with zipfile.ZipFile(filename) as myzip:
                with myzip.open('annotations.xml') as myfile:
                    tree = ET.parse(myfile)
        else:
            tree = ET.parse(filename)
        
        root = tree.getroot()
        assert root.tag == "annotations"
        ann_node = root
        self.original_width = self.original_height = 0
        for el in ann_node.findall('./meta/task/original_size/width'):
            self.original_width = int(el.text)
        for el in ann_node.findall('./meta/task/original_size/height'):
            self.original_height = int(el.text)
        for el in ann_node.findall('./meta/idealfile'):
            self.idealfile = el.text
        #
        label_counter = 0
        for labels_node in ann_node.iter('labels'):
            for label_el in labels_node.iter('label'):
                for el in label_el.iter('name'):
                    self.labels[el.text] = label_counter
                    label_counter += 1
        #
        for track_node in ann_node.iter('track'):
            id = track_node.attrib.get("id", track_node.attrib.get("unique_id"))  ## GVNC
            track = Track(self.frame_poses,
                          id,
                          track_node.attrib.get("label",""),
                          track_node.attrib.get("source","") == "auto"
                          )
            for ann_polygon in track_node.iter('box'):
                pos = Pos(track=track, frame_no=int(ann_polygon.attrib["frame"]),
                          is_outside=bool(int(ann_polygon.attrib["outside"])),
                          is_occluded=bool(int(ann_polygon.attrib["occluded"])),
                          is_keyframe=bool(int(ann_polygon.attrib["keyframe"])), x1=float(ann_polygon.attrib["xtl"]),
                          y1=float(ann_polygon.attrib["ytl"]), x2=float(ann_polygon.attrib["xbr"]),
                          y2=float(ann_polygon.attrib["ybr"]))
                for a in ann_polygon.iter('attribute'):
                    pos.attributes[a.attrib["name"]] += [a.text]
                track.add_pos(pos)
            if track.frames:
                assert id not in self.tracks, f"Error: track {id} is found more then once in {str(filename)}"
                self.tracks[id] = track

    def revise_keyframe_flags(self, add_keypoints=False, remove_keypoins=False, pos_tolerance = None):
        for track in self.tracks.values():
            if add_keypoints or remove_keypoins:
                track.revise_keyframe_flags(add_key=add_keypoints, remove_key=remove_keypoins, pos_tolerance=pos_tolerance)

    def imprint_rects(self, image, image_scale, frame_no, selected_id):
            annotation_scale = max(round(1./image_scale), 1)
            x_scale = image.shape[1]/self.original_width if self.original_width else 1
            y_scale = image.shape[0]/self.original_height if self.original_height else 1
            for pos in self.get_frame_poses(frame_no):
                is_selected = pos.track.id == selected_id
                color_val = 255 if is_selected else 96
                color = (0, color_val, 0) if pos.track.is_auto else (color_val, 0, 0)
                thickness = 3 if pos.is_keyframe and not pos.track.is_auto else 2
                if pos.is_outside or pos.is_occluded:
                    color = tuple((c // 2 for c in color))
                    if pos.is_outside:
                        thickness = 0
                cv2.rectangle(image, (round(pos.x1*x_scale), round(pos.y1*y_scale)), (round(pos.x2*x_scale), round(pos.y2*y_scale)), color=color, thickness=thickness*annotation_scale)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                if pos.track.is_auto:
                    y = min(image.shape[0]-5*y_scale, (pos.y2*y_scale+15*y_scale))
                else:
                    y = max(15*y_scale, (pos.y1 * y_scale - 5*y_scale))
                cv2.putText(image, f"{pos.track.id} {pos.track.label}", (round(pos.x1*x_scale), round(y)), font, fontScale=font_scale/image_scale, color=color, thickness=1*annotation_scale)
