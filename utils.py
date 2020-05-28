from skimage.external import tifffile

from pylab import rcParams
rcParams['figure.figsize'] = (10, 10)

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import skimage.util
from skimage.filters import gaussian
from skimage.util import img_as_ubyte
import struct
from collections import deque
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2


file_header = """MTrackJ 1.5.1 Data File
Displaying true true true 1 2 0 0 100 10 0 0 0 2 1 12 0 true false false false
Assembly 1 FF0000
Cluster 1 FF0000
"""


class Track:
    def __init__(self, d):
        d.pop(), d.pop(), d.pop()
        self.x, self.y, self.length, self.frame_begin = d.pop(), d.pop(), int(d.pop()), int(d.pop())
        self.t1 = [d.pop() for _ in range(self.length)]
        self.xs = [d.pop() for _ in range(self.length)]
        self.ys = [d.pop() for _ in range(self.length)]


def read_tracks_from_gmi_file(file_path):
    with open(file_path, 'rb') as file:
        t = file.read()
    number_of_floats = len(t) // 4
    whole_data = struct.unpack('f' * number_of_floats, t)
    d = deque(whole_data[::-1])
    all_tracks = []
    while d:
        all_tracks.append(Track(d))
    return all_tracks


class GMVDataRead:
    ROWS = 512
    COLS = 512
    frameHeaderDt = np.dtype(
        [('leftx', 'i4'), ('width', 'i4'), ('topy', 'i4'), ('height', 'i4'), ('exp_in_ms', 'i4'), ('fr_size', 'i4'),
         ('fr_time', 'i4'), ('x_nm_pixels', 'f4'), ('y_nm_pixels', 'f4'), ('igain', 'u1'), ('vgain', 'u1'),
         ('bit_pix', 'u1'), ('bin', 'u1'), ('byte_info', 'u1'), ('add_info', 'u1'), ('laser_power', 'i2'),
         ('temperature', 'i2'), ('illum_time', 'i2')])

    def __init__(self, path_to_gmv, path_to_gmi=None):
        self._read_images_list(path_to_gmv)
        if path_to_gmi is not None:
            self._get_coords_from_gmi(path_to_gmi)
            #             self._draw_tracks_from_gmi_on_images_list()
            self._save_data_to_mtrackj_format()

    def save_data(self, tiff_path, mdf_path):
        tifffile.imsave(tiff_path, np.array(self.images_list))
        with open(mdf_path, 'w') as f:
            f.write(self.text_form)

    def _save_data_to_mtrackj_format(self):
        self.text_form = file_header
        for track_number, track in enumerate(self.all_tracks):
            self.text_form += f"Track {track_number + 1} FF0000 true\n"
            xs = track.xs
            ys = track.ys
            for i in range(len(ys)):
                p1 = xs.astype('int')[i], ys.astype('int')[i]
                self.text_form += f"Point {i + 1} {p1[0]} {p1[1]} 1.0 {track.frame_begin + i + 1} 1.0\n"
        self.text_form += 'End of MTrackJ Data File\n'

    def _read_images_list(self, file_path):
        self.frameHeaders = np.array([], dtype=self.frameHeaderDt)
        self.images_list = []
        with open(file_path, 'rb') as fd:
            while True:
                f1 = np.fromfile(fd, dtype=self.frameHeaderDt, count=1)
                if f1.nbytes < 48:
                    print('finished reading')
                    break
                self.frameHeaders = np.resize(self.frameHeaders, self.frameHeaders.size + 1)
                self.frameHeaders[-1] = f1
                f2 = np.fromfile(fd, dtype=np.uint16, count=self.ROWS * self.COLS)
                if f2.nbytes < f1['fr_size']:  # some bad cases??????
                    break
                #     if frameIndex >= readFramesFrom - 1:

                im = f2.reshape((self.ROWS, self.COLS))  # notice row, column format
                #                 print(im.max(), im.min())
                #                 im = im / 3.5                   # very strange processing
                #                 im[np.where(im > 255)] = 255
                im = ((im - im.min()) / (im.max() - im.min())) * 255.0
                im = np.uint8(im)
                self.images_list.append(im)

    def _get_coords_from_gmi(self, file_path):
        """
        Get coordinates (x, y) from gmi file, 
        then normalize it. Get it to pixel coordinates. 
        Get 
        """
        self.all_tracks = read_tracks_from_gmi_file(file_path)
        x_nm_pixels = self.frameHeaders[0]['x_nm_pixels']  # assume equal for all
        y_nm_pixels = self.frameHeaders[0]['y_nm_pixels']
        for track_number, track in enumerate(self.all_tracks):
            # assume x_nm_pixels doesn't change during file!!!
            pos_x = track.x * x_nm_pixels
            pos_y = track.y * y_nm_pixels
            xs = (pos_x + np.array(track.xs) * 1000) / x_nm_pixels
            ys = (pos_y + np.array(track.ys) * 1000) / y_nm_pixels
            self.all_tracks[track_number].xs = xs
            self.all_tracks[track_number].ys = ys

    def get_velocities_normalized(self):
        """
        can be used only after _get_coords_from_gmi!!!
        """
        time_frame = np.array([el['fr_time'] for el in self.frameHeaders])
        self.time_per_frame = np.diff(time_frame)  # how long each frame lasted
        self.normalized_velocities = []
        for track_number, track in enumerate(self.all_tracks):
            path_x = np.diff(track.xs)
            path_y = np.diff(track.ys)
            self.normalized_velocities.append((
                path_x / self.time_per_frame[track.frame_begin:track.frame_begin + len(path_x)],
                path_y / self.time_per_frame[track.frame_begin:track.frame_begin + len(path_y)])
            )
        return self.normalized_velocities

    def _draw_tracks_from_gmi_on_images_list(self):
        self.normalized_velocities = []
        for track_number, track in enumerate(self.all_tracks):
            # assume x_nm_pixels doesn't change during file!!!
            xs = track.xs
            ys = track.ys
            for frame_number in range(len(self.images_list)):
                #                 self.images_list[frame_number][ys.astype('int'), xs.astype('int')] = 125
                #                 self.images_list[frame_number][ys.astype('int')[0], xs.astype('int')[0]] = 255
                #                 self.images_list[frame_number][ys.astype('int')-1, xs.astype('int')-1] = 125
                #                 self.images_list[frame_number][ys.astype('int')+1, xs.astype('int')+1] = 125
                #                 self.images_list[frame_number][ys.astype('int')-1, xs.astype('int')+1] = 125
                #                 self.images_list[frame_number][ys.astype('int')+1, xs.astype('int')-1] = 125
                for i in range(len(ys) - 1):
                    p1 = xs.astype('int')[i], ys.astype('int')[i]
                    p2 = xs.astype('int')[i + 1], ys.astype('int')[i + 1]
                    cv2.arrowedLine(self.images_list[frame_number], p1, p2, color=255)

    def save_images_list_to_pdf(self, path_to_pdf):
        for ind, image in enumerate(self.images_list):
            w, h = plt.figaspect(image.shape[0] / image.shape[1])
            fig = plt.figure(num=ind, figsize=(w, h))
            fig.set_dpi(400)
            plt.imshow(image, cmap='gray')

        pdf = PdfPages(path_to_pdf)
        for fig_num in plt.get_fignums():
            pdf.savefig(fig_num)
        pdf.close()
        plt.close('all')