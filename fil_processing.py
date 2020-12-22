import numpy as np
import pickle as pkl
# import tifffile
from skimage.external import tifffile
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.optimize import linear_sum_assignment
import cv2
import pathlib
import argparse
from math import fabs
from numba import jit
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
# from fire import Cluster
# from typing import List
from collections import defaultdict, namedtuple
from skimage.draw import rectangle_perimeter, ellipse_perimeter

FIRE_NUM = 5
INF_DISTANCE = 10_000
MIN_FIL_LEN_BF = 3


def find_tip_in_filament(im, i, j, was_here, padding):
    was_here[i][j] = 255
    found = 0
    for t in range(-1, 2):
        for k in range(-1, 2):
            if (t != 0) or (k != 0):
                if (i + t >= padding) and (i + t < im.shape[0] - padding) and (j + k >= padding) and (
                        j + k < im.shape[1] - padding):
                    if (im[i + t][j + k] == 255) and (was_here[i + t][j + k] == 0):
                        found += 1
                        return find_tip_in_filament(im, i + t, j + k, was_here, padding)
    if found == 0 or found == 1:
        # print("Found tip!")
        return i, j


def follow_filament(im, i, j, fil_list, was_here, num, padding=0, is_first=True):
    """
    Want to make sure that filament first anf last coords are tips
    """
    if is_first:
        i, j = find_tip_in_filament(im, i, j, was_here.copy(), padding)
    # print(i, j)
    was_here[i][j] = 255
    fil_list[num].append([i, j])
    found = 0
    for t in range(-1, 2):
        for k in range(-1, 2):
            if (t != 0) or (k != 0):
                if (i + t >= padding) and (i + t < im.shape[0] - padding) and (j + k >= padding) and (
                        j + k < im.shape[1] - padding):
                    if (im[i + t][j + k] == 255) and (was_here[i + t][j + k] == 0):
                        found += 1
                        follow_filament(im, i + t, j + k, fil_list, was_here, num, is_first=False)

    if found > 3:
        print('very bad')


def find_filaments(im):
    fil_list = []
    sh = im.shape
    was_here = np.zeros_like(im)
    number_of_filaments = 0
    for i in range(sh[0]):
        for j in range(sh[1]):
            if im[i][j] == 255.0 and not was_here[i][j]:
                fil_list.append([])
                follow_filament(im, i, j, fil_list, was_here, number_of_filaments, is_first=True)
                number_of_filaments += 1
    return fil_list


class TipTrack:
    def __init__(self, first_frame, x, y):
        self.xs = x
        self.ys = y
        self.frame_begin = first_frame


class Tracks:
    # def __init__(self):
    #     self.paths = []
    #     self.tips = []
    def __init__(self, frame=None, prediction_type='trivial'):
        self.tips = []
        if frame is not None:
            self.paths = [Path(frame.num, filament, prediction_type) for filament in frame.filaments]
        else:
            self.paths = []

    def _create_tips_from__paths(self):
        self.tips = []  # clear all previous?
        for path in self.paths:
            xs, ys = [], []
            for filament in path.filament_path:
                xs.append(filament.xs[0])  # choose what point to save
                ys.append(filament.ys[0])
            self.tips.append(TipTrack(path.first_frame_num, np.array(ys), np.array(xs)))

    def _save_data_to_mtrackj_format_old(self, mdf_path):
        self._create_tips_from__paths()
        # can save only one point from filament, that's why use tip class.
        from utils import file_header
        self.text_form = file_header
        for track_number, track in enumerate(self.tips):
            self.text_form += f"Track {track_number + 1} FF0000 true\n"
            xs = track.xs
            ys = track.ys
            for i in range(len(ys)):
                p1 = xs.astype('int')[i], ys.astype('int')[i]
                self.text_form += f"Point {i + 1} {p1[0]} {p1[1]} 1.0 {track.frame_begin + i + 1} 1.0\n"
        self.text_form += 'End of MTrackJ Data File\n'

        with open(mdf_path, 'w') as f:
            f.write(self.text_form)

    def _save_data_to_mtrackj_format_new(self, mdf_path):
        self._create_tips_from__paths()
        # can save only one point from filament, that's why use tip class.
        from utils import file_header
        self.text_form = file_header
        for track_number, track in enumerate(self.paths):
            # for track_number, track in enumerate(self.tips):
            self.text_form += f"Track {track_number + 1} 00FFFF true\n"
            # xs = track.xs
            # ys = track.ys
            for i, filament in enumerate(track.filament_path):
                # p1 = xs.astype('int')[i], ys.astype('int')[i]
                self.text_form += f"Point {i + 1} {filament.tail[1]} {filament.tail[0]} 1.0 {track.first_frame_num + i + 1} 1.0\n"
        self.text_form += 'End of MTrackJ Data File\n'

        with open(mdf_path, 'w') as f:
            f.write(self.text_form)


class Filament:
    def __init__(self, list_coords, number_of_tips=None):
        self.coords = np.array(list_coords)  # can be float, but x, y coords are int
        self.xs = np.around(self.coords[:, 0]).astype('int')
        self.ys = np.around(self.coords[:, 1]).astype('int')
        self.cm = np.array([np.mean(self.xs), np.mean(self.ys)])  # center mass
        self.center_x = int(round(self.cm[0]))
        self.center_y = int(round(self.cm[1]))
        ### do i really should calculate length with ints, not floats!!!???
        self.length = np.sum(np.sqrt(np.diff(self.xs) ** 2 + np.diff(self.ys) ** 2))
        if number_of_tips is None:
            self.number_of_tips = self.get_number_of_tips()
        else:
            self.number_of_tips = number_of_tips

    @property
    def head(self):
        return self.coords[0]

    @property
    def tail(self):
        return self.coords[-1]

    def get_number_of_tips(self):
        ### really bad ####
        ### TODO: fiiiiix ###
        img = np.zeros(shape=(512, 512))
        img[self.xs, self.ys] = 1
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ])
        d = convolve2d(img, kernel)
        return len(np.where(d == 11)[0])

    @staticmethod
    def fils_distance_cm(filament_1, filament_2):
        # TODO: very simple distance. make better!
        return np.sqrt(((filament_1.cm - filament_2.cm) ** 2).sum())

    @staticmethod
    def fils_distance_cm_length(filament_1, filament_2, lambda_=0.7):
        return lambda_ * Filament.fils_distance_cm(filament_1, filament_2) + \
               (1 - lambda_) * np.sqrt(((filament_1.length - filament_2.length)**2))

    @staticmethod
    def get_long_short_filament(filament_1, filament_2):
        short_fil, long_fil = filament_1, filament_2
        if len(short_fil.xs) > len(long_fil.xs):
            short_fil, long_fil = long_fil, short_fil
        # coords -- n+1, where n -- is number of vectors!!!
        short_len, long_len = len(short_fil.xs), len(long_fil.xs)
        long_contour = long_fil.coords
        short_contour = short_fil.coords
        return long_contour, long_len, short_contour, short_len

    @staticmethod
    def fils_distance_fast(filament_1, filament_2, fil_direction=1):
        fil_direction = 1
        # Why do I need here filament direction
        # TODO: distance like in FAST, Aksel!

        long_contour, long_len, short_contour, short_len = Filament.get_long_short_filament(filament_1,
                                                                                            filament_2)
        # np.sqrt(np.sum((np.diff(long_contour, axis=0) - np.diff(short_contour, axis=0)) ** 2))
        multiplicate_measures = long_len - short_len + 1
        distance_score = 0  # from aksel!!
        for i in range(multiplicate_measures):
            long_short_diff = long_contour[i:i + short_len, :][::fil_direction] - short_contour
            distance_length = np.mean(np.sqrt(np.sum(long_short_diff ** 2, axis=1)))
            distance_score += distance_length
        distance_score /= multiplicate_measures
        return distance_score

    @staticmethod
    # @jit(nopython=True)
    def overlap_score_fast_numba(long_contour, long_len, short_contour, short_len, fil_direction=1):
        multiplicate_measures = long_len - short_len + 1
        overlap_score = 0  # from aksel!!
        # need to set short_len - 1 --- number of vectors!
        for i in range(multiplicate_measures):
            long_short_product = np.diff(
                long_contour[i:i + short_len, :][::fil_direction],
                axis=0
            ) * \
                                 np.diff(short_contour, axis=0)
            scalar_product = np.sum(long_short_product)
            overlap_score += scalar_product
        overlap_score /= multiplicate_measures
        overlap_score /= (short_len - 1)
        return overlap_score

    @staticmethod
    def overlap_score_fast(filament_1, filament_2, fil_direction=1):
        fil_direction = 1
        # Why do I need here filament direction
        # TODO: distance like in FAST, Aksel!

        long_contour, long_len, short_contour, short_len = Filament.get_long_short_filament(filament_1,
                                                                                            filament_2)
        return Filament.overlap_score_fast_numba(long_contour, long_len, short_contour,
                                                 short_len, fil_direction=fil_direction)

    @staticmethod
    def draw_filaments(filaments, img=None, to_draw=True, path_to_save=None):
        plt.figure(figsize=(5, 5), dpi=100)
        if not isinstance(filaments, list):
            filaments = [filaments]
        if img is None:
            img = np.zeros(shape=(512, 512, 3))
        len_fils = len(filaments)
        for ind, filament in enumerate(filaments):
            # g_color = np.random.randint(1, 256)
            # b_color = np.random.randint(1, 256)
            g_color = 256 / len_fils * (ind + 1)
            b_color = 256 - 256 / len_fils * (ind + 1)
            img[filament.xs, filament.ys, 1] = g_color
            img[filament.xs, filament.ys, 2] = b_color
            print(g_color, b_color)
            img[filament.center_x, filament.center_y, 0] = 255

        if path_to_save is not None:
            plt.imsave(path_to_save, img / img.max())

        if to_draw:
            plt.imshow(img, interpolation='bicubic')
            plt.show()
        else:
            return img


def init_kf():
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([
        [1., 0, 1, 0],
        [0., 1, 0, 1],
        [0., 0, 1, 0],
        [0., 0, 0, 1]
    ])

    kf.H = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]]
    )

    kf.P[:2, :2] *= 20
    kf.P[2:, 2:] *= 300
    kf.R = kf.R * 20

    Q_std = 5.0
    q = Q_discrete_white_noise(dim=2, dt=1, var=Q_std**2)
    kf.Q[0,0] = q[0,0]
    kf.Q[1,1] = q[0,0]
    kf.Q[2,2] = q[1,1]
    kf.Q[3,3] = q[1,1]
    kf.Q[0,2] = q[0,1]
    kf.Q[2,0] = q[0,1]
    kf.Q[1,3] = q[0,1]
    kf.Q[3,1] = q[0,1]

    return kf


class Dummy_Filament:
    """
    This dummy filament is used for Kalman filter and cm distance function
    Can't use this Kalman filter with not cm distance functions
    """
    def __init__(self, c_x, c_y):
        self.cm = np.array([c_x, c_y])


class Path:
    def __init__(self, first_frame_num, filament : Filament, prediction_type='trivial'):
        """
        prediction type can be trivial: return last elem
        or using kalman filter
        """
        self.filament_path = [filament]
        self.mean_length = filament.length

        self.first_frame_num = first_frame_num
        self.is_finished = False
        self.next_center_velocity = np.array([[0, 0]])
        self.prediction_type = prediction_type
        self.next_filament = None

        self.kf = init_kf()
        self.kf.x[:2] = filament.cm[:, np.newaxis]
        self.predicted_history = []

    @property
    def last_filament(self):
        return self.filament_path[-1]

    def draw_link(self):
        if len(self.filament_path) < 2:
            return
        img = np.zeros(shape=(512, 512, 3))
        img[self.filament_path[-1].xs, self.filament_path[-1].ys, 1:2] = 255
        img[self.filament_path[-2].xs, self.filament_path[-2].ys, 2:3] = 255
        img[self.filament_path[-2].center_x, self.filament_path[-2].center_y, 0] = 255
        img[self.filament_path[-1].center_x, self.filament_path[-1].center_y, 0] = 255
        img[self.next_filament_predicted.xs, self.next_filament_predicted.ys, 1:2] = 100
        img = cv2.arrowedLine(img,
                              (self.filament_path[-2].center_y, self.filament_path[-2].center_x),
                              (self.filament_path[-1].center_y, self.filament_path[-1].center_x),
                              (255, 0, 0), 1)
        img = cv2.arrowedLine(img,
                              (self.filament_path[-1].center_y, self.filament_path[-1].center_x),
                              (self.next_filament_predicted.center_y, self.next_filament_predicted.center_x),
                              (255, 0, 0), 1)
        plt.imshow(img, interpolation='bicubic')
        plt.show()
        print('draw')

    @property
    def next_filament_predicted(self):
        return self.next_filament
        # new_filament_coords = self.filament_path[-1].coords + self.next_center_velocity
        # return Filament(new_filament_coords.tolist(),
        #                 self.filament_path[-1].number_of_tips)

    def predict_next_state(self):
        self.kf.predict()
        self.predicted_history.append([self.kf.x, self.kf.P, self.filament_path[-1]])

        if self.prediction_type == 'trivial':
            self.next_filament = self.filament_path[-1]
        else:
            if len(self.predicted_history) > 5:
                self.next_filament = Dummy_Filament(self.kf.x[0, 0], self.kf.x[1, 0])
            else:
                self.next_filament = self.filament_path[-1]

    def check_gate(self):
        """
        if using Kalman filter, check gate must be called after prediction step was made
        """
        if True or self.prediction_type == 'trivial' or len(self.filament_path) <= 5:
            xmin, xmax = self.filament_path[-1].cm[0] - self.mean_length, \
                         self.filament_path[-1].cm[0] + self.mean_length
            ymin, ymax = self.filament_path[-1].cm[1] - self.mean_length,\
                         self.filament_path[-1].cm[1] + self.mean_length
        else:
            x, y, v_x, v_y = self.kf.x.flatten()
            xmin, xmax = x - 3 * v_x, x + 3 * v_x
            ymin, ymax = y - 3 * v_y, y + 3 * v_y

        return xmin, xmax, ymin, ymax

    def add(self, filament):
        self.filament_path.append(filament)
        self.kf.update(np.reshape(filament.cm, [2, 1]))
        # TODO: fix mean calc, make more fast
        self.mean_length = sum([fil.length for fil in self.filament_path]) / len(self.filament_path)


def get_biggest_chunk_array(fil):
    if len(fil) <= MIN_FIL_LEN_BF:
        fil = None
        return fil

    branches = np.where(np.abs(np.diff(np.array(fil), axis=0)) > 1)[0]
    if len(branches) > 0:
        possible_cuts = np.concatenate([[0], branches + 1, [len(fil)]])
        max_cut = np.argmax(np.diff(possible_cuts))
        fil = fil[possible_cuts[max_cut]:possible_cuts[max_cut + 1]]

    if len(fil) <= MIN_FIL_LEN_BF:
        fil = None
    return fil


def get_biggest_chunk_array_2(fil):
    branches = np.where(np.abs(np.diff(np.array(fil), axis=0)) > 1)[0]
    if len(branches) == 2:
        branch_1 = fil[0:branches[0] + 1]
        branch_2 = fil[branches[0] + 1:]
        dist_between_two_chucnks = np.sqrt(((branch_1[-1] - branch_2[0]) ** 2).sum())
        if dist_between_two_chucnks < 10:
            return fil

    if len(branches) > 0:
        # print('bad fil')
        possible_cuts = np.concatenate([[0], branches + 1, [len(fil)]])
        max_cut = np.argmax(np.diff(possible_cuts))
        fil = fil[possible_cuts[max_cut]:possible_cuts[max_cut + 1]]
        # print(possible_cuts[max_cut+1] - possible_cuts[max_cut])
        # all_branches.append(branches)
    if len(fil) <= 3:
        fil = None
    return fil


def check_filament_length_ratio(max_length, min_length, ratio_big_filaments,
                                ratio_small_filaments, len_small_filaments,
                                process_length_small_filaments_type):
    if process_length_small_filaments_type == 'simple':
        if max_length / min_length > ratio_big_filaments:
            return INF_DISTANCE
    elif process_length_small_filaments_type == 'no_processing':
        if max_length > len_small_filaments and max_length / min_length > ratio_big_filaments:
            return INF_DISTANCE
    elif process_length_small_filaments_type == 'processing':
        if max_length > len_small_filaments and max_length / min_length > ratio_big_filaments:
            return INF_DISTANCE
        if max_length < len_small_filaments and max_length / min_length > ratio_small_filaments:
            return INF_DISTANCE
    else:
        raise NotImplementedError

    return -1


# TODO: change this later as was discussed!!!
def process_filaments_simple(filament_list):
    # all_branches = []
    filaments = []

    # print('fil_length: ***')

    filament_lengths = [len(fil) for fil in filament_list]
    # unique, counts = np.unique(filament_lengths, return_counts=True)
    # plt.hist(filament_lengths, bins=len(unique))
    # plt.show()

    for fil in filament_list:
        fil = get_biggest_chunk_array(fil)
        if fil is None:
            continue

        # if len(fil) > 4 and len(fil) < 10:
        #     print('aaa')

        filament = Filament(fil)
        # if filament.length < 5:
        #     continue
        # how to get two points!???
        # if filament.number_of_tips <= 2:
        filaments.append(filament)
    return filaments


def process_filaments_no_proccesing(filament_list):
    filaments = [Filament(fil) for fil in filament_list]
    return filaments


def choose_process_func(processing_frame: str):
    if processing_frame == 'no processing':
        return process_filaments_no_proccesing
    elif processing_frame == 'simple':
        return process_filaments_simple
    elif processing_frame == 'fire':
        raise NotImplementedError
    else:
        raise NotImplementedError


def choose_add_fil_func(use_overlap_score):
    if use_overlap_score:
        return add_filament_v1
    else:
        return add_filament_v2


class Frame:
    def __init__(self, img, frame_num, processing_frame):
        """
        img is binary image.
        """
        self.num = frame_num
        ### TODO: work with more that two points filaments!!!
        process_filaments = choose_process_func(processing_frame)
        self.filaments = process_filaments(find_filaments(img))
        self.links = []

    @staticmethod
    def draw_frame(frame):
        img = np.zeros(512, 512, 3)
        for filament in frame.filaments:
            img = Filament.draw_filament(filament, img, to_draw=False)
        return img


def get_path_fil_length(path, path_filament_len_type):
    path_fils_length = [fil.length for fil in path.filament_path]
    if path_filament_len_type == 'last':
        filament_len = path_fils_length[-1]
    elif path_filament_len_type == 'mean':
        filament_len = sum(path_fils_length) / len(path_fils_length)
    elif path_filament_len_type == 'weighted_mean':
        # den_mean = 0
        # weighted_mean = 0
        # for ind_mean, pfl in enumerate(path_fils_length):
        #     den_mean += ind_mean + 1
        #     weighted_mean += (ind_mean + 1) * pfl
        # weighted_mean = weighted_mean / den_mean
        # mean_path_len = weighted_mean
        raise NotImplementedError
    else:
        raise NotImplementedError

    return filament_len


def choose_distance(type_):
    if type_ == 'cm':
        return Filament.fils_distance_cm
    elif type_ == 'fast_like':
        return Filament.fils_distance_fast
    elif type_ == 'cm_length':
        return Filament.fils_distance_cm_length


def choose_gate_func(type_):
    if type_ == 'constant':
        return constant_gate
    elif type_ == 'motion':
        return motion_gate
    elif type_ == 'motion_kalman':
        return motion_kalman
    elif type_ == 'motion_dense':
        return motion_dense_gate


def constant_gate(path: Path, distance_to_fils, constant=20):
    new_distance_to_fils = []
    for dist in distance_to_fils:
        if dist > constant:
            dist = INF_DISTANCE
        new_distance_to_fils.append(dist)
    return new_distance_to_fils


def motion_gate(path: Path, distance_to_fils):
    if len(path.filament_path) < 4:
        return constant_gate(path, distance_to_fils)

    # TODO: optimize calculations, maybe add more weight to last measurements
    movement_history = []
    prev_pos = path.filament_path[0].cm
    for fil in path.filament_path[1:]:
        movement_history.append(prev_pos - fil.cm)
        prev_pos = fil.cm

    found_displacement = np.mean(np.sqrt(np.sum(np.array(movement_history) ** 2, axis=1)))

    # displacements_filaments = np.sqrt(np.sum(np.array(movement_history) ** 2, axis=1))
    #
    # den_mean = 0
    # weighted_mean = 0
    # for ind_mean, pfl in enumerate(displacements_filaments):
    #     den_mean += ind_mean + 1
    #     weighted_mean += (ind_mean + 1) * pfl
    # weighted_mean = weighted_mean / den_mean
    # found_displacement = weighted_mean

    return constant_gate(path, distance_to_fils, constant=5*found_displacement)


# Add variations to this parameter
def motion_kalman(path: Path, distance_to_fils, num_of_sigmas=5):
    if len(path.filament_path) < 4:
        return constant_gate(path, distance_to_fils)

    found_displacement = np.sqrt(np.sum(path.kf.x[2:, 0] ** 2))
    sigma2_x, sigma2_y = path.kf.P[2, 2], path.kf.P[3, 3]
    sigma = np.sqrt(sigma2_x + sigma2_y)
    return constant_gate(path, distance_to_fils, constant=found_displacement + num_of_sigmas * sigma)


def motion_dense_gate(path: Path, distance_to_fils):
    pass


class FireTrack:
    def __init__(self, cluster, cluster_num, branch):
        self.cluster = cluster
        self.cluster_num = cluster_num
        self.branch = branch
        self.frames_filaments = defaultdict(list)

    def add_filament(self, filament, min_dist, frame_num):
        self.frames_filaments[frame_num].append([min_dist, filament])


def connect_two_filaments(first_filament, second_filament):
    from fire import calc_min_distance_bt_four_points

    min_ind, min_dist = calc_min_distance_bt_four_points(
        first_filament.coords[0], first_filament.coords[-1],
        second_filament.coords[0], second_filament.coords[-1]
    )

    ### TODO: fix some constant!
    print('min_dist_for_connnecion: ', min_dist)
    if min_dist < 10:
        if min_ind == 0:
            # start, start
            fils_coords = [
                *[coord for coord in first_filament.coords.tolist()[::-1]],
                *[coord for coord in second_filament.coords.tolist()]
            ]
        if min_ind == 1:
            # start, end
            fils_coords = [
                *[coord for coord in second_filament.coords.tolist()],
                *[coord for coord in first_filament.coords.tolist()]
            ]
        if min_ind == 2:
            # end, start
            fils_coords = [
                *[coord for coord in first_filament.coords.tolist()],
                *[coord for coord in second_filament.coords.tolist()]
            ]
        if min_ind == 3:
            # end, end
            fils_coords = [
                *[coord for coord in second_filament.coords.tolist()],
                *[coord for coord in first_filament.coords.tolist()[::-1]]
            ]

        new_filament = Filament(fils_coords)
    else:
        # TODO: fix what to do
        # was_nly_one = True
        # lengths_filaments.append(v[0][1].length)
        new_filament = None
    return new_filament


def find_closest_ind_in_array_to_point(points, close_point):
    min_ind = -1
    min_dist = 10_000
    for ind, point in enumerate(points):
        dist = np.sqrt(((point - close_point) ** 2).sum())
        if dist < min_dist:
            min_dist = dist
            min_ind = ind
    return min_ind


def find_tail(path):
    """
    Modifies path variable
    """
    if len(path.filament_path) < 2:
        return False

    second_filament = path.filament_path[-1]
    first_filament = path.filament_path[-2]
    head = second_filament.head
    tail = second_filament.tail
    min_head = min([np.sqrt(((point - head) ** 2).sum()) for point in first_filament.coords])
    min_tail = min([np.sqrt(((point - tail) ** 2).sum()) for point in first_filament.coords])
    if min_tail > min_head:
        path.filament_path = [
            *path.filament_path[:-1],
            # Filament(first_filament.coords[::-1], number_of_tips=first_filament.number_of_tips),
            Filament(second_filament.coords[::-1], number_of_tips=second_filament.number_of_tips)
        ]

    # check first filament
    if len(path.filament_path) == 2:
        second_filament = path.filament_path[-2]
        first_filament = path.filament_path[-1]
        head = second_filament.head
        tail = second_filament.tail
        min_head = min([np.sqrt(((point - head) ** 2).sum()) for point in first_filament.coords])
        min_tail = min([np.sqrt(((point - tail) ** 2).sum()) for point in first_filament.coords])
        if min_tail < min_head:
            path.filament_path = [
                Filament(second_filament.coords[::-1], number_of_tips=second_filament.number_of_tips),
                Filament(first_filament.coords, number_of_tips=first_filament.number_of_tips)
            ]

    return True


def add_filament_v1(path, found_filament):
    """
    For the first time calc with func find_tail
    After that, use only overlap score
    """
    ##### GET OVERLAP
    prev_filament = path.filament_path[-1]
    if Filament.overlap_score_fast(prev_filament, found_filament) < 0:
        found_filament = Filament(found_filament.coords[::-1],
                                  number_of_tips=found_filament.number_of_tips)
    #########
    path.add(found_filament)
    if len(path.filament_path) == 2:
        find_tail(path)


def add_filament_v2(path, found_filament):
    """
    Don't use distance from FIRE,
    calc tail and head each time
    """
    path.add(found_filament)
    find_tail(path)


def reconstruct_filament(prev_filament: Filament, future_filament: Filament,
                         mean_filament_length: float, ft: FireTrack):
    """
    If we have prev and next filament, then take mean from their cms, and expand in two directions on firetrack
    If only last, take last point in firetrack and expand until length
    """
    from scipy.interpolate import splprep, splev

    x, y = zip(*ft.branch.points)
    tck, u = splprep([np.array(x), np.array(y)], s=0)
    points_to_construct_filament = splev(np.linspace(0, 1, 100), tck)
    points_to_construct_filament = [np.array(el) for el in zip(*points_to_construct_filament)]

    if future_filament is None:
        ### last filament
        left_point = len(points_to_construct_filament) - 1
        length_of_new_filament = 0
        coords_for_new_filament = [points_to_construct_filament[-1]]
        filled_branch_left = False
        while length_of_new_filament < mean_filament_length and not filled_branch_left:
            grown_length_left = 0
            if left_point > 0:
                left_point = left_point - 1
                coords_for_new_filament.insert(0, points_to_construct_filament[left_point])
                grown_length_left = np.sqrt(((coords_for_new_filament[0] -
                                              coords_for_new_filament[1]) ** 2).sum())
                if grown_length_left == 0:
                    # TODO: fix!
                    continue
            else:
                filled_branch_left = True

            # print(grown_length_left)
            length_of_new_filament += grown_length_left
    elif prev_filament is None:
        left_point = 0
        length_of_new_filament = 0
        coords_for_new_filament = [points_to_construct_filament[0]]
        filled_branch_left = False
        while length_of_new_filament < mean_filament_length and not filled_branch_left:
            grown_length_left = 0
            if left_point < len(points_to_construct_filament) - 1:
                left_point = left_point + 1
                coords_for_new_filament.append(points_to_construct_filament[left_point])
                grown_length_left = np.sqrt(((coords_for_new_filament[-1] -
                                              coords_for_new_filament[-2]) ** 2).sum())
                if grown_length_left == 0:
                    # TODO: fix!
                    continue
            else:
                filled_branch_left = True

            # print(grown_length_left)
            length_of_new_filament += grown_length_left
    else:
        new_filament_cm = (prev_filament.cm + future_filament.cm) / 2

        min_ind = find_closest_ind_in_array_to_point(points_to_construct_filament, new_filament_cm)

        left_point = min_ind
        right_point = min_ind
        length_of_new_filament = 0
        coords_for_new_filament = [points_to_construct_filament[min_ind]]
        filled_branch_left = False
        filled_branch_right = False
        while length_of_new_filament < mean_filament_length and \
                not filled_branch_right and not filled_branch_left:
            grown_length_left = 0
            grown_length_right = 0
            if left_point > 0:
                left_point = left_point - 1
                coords_for_new_filament.insert(0, points_to_construct_filament[left_point])
                grown_length_left = np.sqrt(((coords_for_new_filament[0] -
                                              coords_for_new_filament[1]) ** 2).sum())
                if grown_length_left == 0:
                    # TODO: fix!
                    continue
            else:
                filled_branch_left = True

            if right_point < len(points_to_construct_filament) - 1:
                right_point = right_point + 1
                coords_for_new_filament.append(points_to_construct_filament[right_point])
                grown_length_right = np.sqrt(((coords_for_new_filament[-1] -
                                               coords_for_new_filament[-2]) ** 2).sum())
                if grown_length_right == 0:
                    # TODO: fix!!!
                    continue
            else:
                filled_branch_right = True

            # print(grown_length_left, grown_length_right)
            length_of_new_filament += (grown_length_right + grown_length_left)

    new_filament = Filament(coords_for_new_filament)
    return new_filament


class Video:
    def __init__(self, list_images,
                 fire_frames,
                 processing_frame: str,
                 path_to_results):
        """
        Assume that all pictures from list_images of equal size
        Maybe should check list_images!!!
        fire_frames: List[List[Cluster]]
        :param list_images: list of [m, n, 3] images. [0] -- binary, [1]=[2]=actual image
        """
        if len(list_images[0].shape) == 2:
            # TODO: check whether it works!
            list_images = [np.expand_dims(img, axis=2) for img in list_images]
        self.width = list_images[0].shape[0]
        self.height = list_images[0].shape[1]
        self.frames = [Frame(img[:, :, 0], ind, processing_frame) for ind, img in enumerate(list_images)]
        self.fire_frames = fire_frames
        self.tracks = Tracks()
        self.loaded_fire = False
        self.path_to_results = path_to_results

    def update_path_to_results(self, path_to_res):
        self.path_to_results = path_to_res

    def visualize_by_frames(self, save_file=None):
        img = np.zeros(shape=[len(self.frames), self.width, self.height, 3])
        for frame_num, frame in enumerate(self.frames):
            for _, filament in enumerate(frame.filaments):
                img[frame_num][filament.xs, filament.ys, 1:3] = 255
                img[frame_num][filament.center_x, filament.center_y, 0] = 255
        if save_file is not None:
            tifffile.imsave(save_file, data=img.astype('uint8'))
        else:
            return img.astype('uint8')

    def visualize_by_frames_2(self, frame2filaments, save_file):
        if len(frame2filaments) == 0:
            return
        img = np.zeros(shape=[len(frame2filaments), self.width, self.height, 3])
        for inf, (frame_num, filaments) in enumerate(frame2filaments.items()):
            for _, filament in enumerate(filaments):
                img[inf][filament.xs, filament.ys, 1:3] = 255
                img[inf][filament.center_x, filament.center_y, 0] = 255
        tifffile.imsave(save_file, data=img.astype('uint8'))

    def visualize_by_frames_gates(self, path_to_file,
                                  save_file=None,
                                  gate_type='kalman_speed_est_square',
                                  draw_all=False):
        img_orig = tifffile.imread(str(path_to_file))
        img = np.zeros(shape=[len(self.frames), self.width, self.height, 3])
        img[:, :, :, 2] = img_orig
        print(img_orig.shape)

        for path in self.tracks.paths:
            print(path.predicted_history)
            for ind, (x, P, fil) in enumerate(path.predicted_history):
                frame_num = path.first_frame_num + ind
                # filament_center = min(int(x[0]), self.width-1), min(int(x[1]), self.height-1)
                filament_center = fil.center_x, fil.center_y
                img[frame_num, filament_center[0], filament_center[1], 0] = 255

                if draw_all or gate_type == 'kalman_speed_est_square':
                    found_displacement = np.sqrt(np.sum(x[2:, 0] ** 2))
                    sigma2_x, sigma2_y = P[2, 2], P[3, 3]
                    sigma = int(np.sqrt(sigma2_x + sigma2_y))
                    constant = (found_displacement + 5 * sigma) // 2
                    start = (filament_center[0] - constant , filament_center[1] - constant)
                    end = (filament_center[0] + constant, filament_center[1] + constant)
                    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape[1:])
                    if draw_all:
                        img[frame_num, rr, cc, 2] = 255

                if draw_all or gate_type == 'kalman_coords_est_square':
                    found_displacement = 0 # np.sqrt(np.sum(x[2:, 0] ** 2))
                    sigma2_x, sigma2_y = P[0, 0], P[1, 1]
                    sigma = int(np.sqrt(sigma2_x + sigma2_y))
                    constant = (found_displacement + 5 * sigma) // 2
                    start = (filament_center[0] - constant , filament_center[1] - constant)
                    end = (filament_center[0] + constant, filament_center[1] + constant)
                    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape[1:])
                    if draw_all:
                        img[frame_num, rr, cc, 0] = 255

                if draw_all or gate_type == 'kalman_ellipsis_est_square':
                    from filterpy.stats import covariance_ellipse
                    # cov = np.array([[P[0, 0], P[2, 0]],
                    #                 [P[0, 2], P[2, 2]]])
                    # mean = (x[0, 0], x[2, 0])
                    # plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)
                    cov = P[0:2, 0:2]
                    ellipse = covariance_ellipse(cov)
                    # angle = np.degrees(ellipse[0])
                    angle = ellipse[0]
                    width = ellipse[1]
                    height = ellipse[2]

                    rr, cc = ellipse_perimeter(
                        fil.center_x, fil.center_y, int(3 * width), int(3 * height),
                        angle, shape=img.shape[1:]
                    )

                    if draw_all:
                        img[frame_num, rr, cc, 1:2] = 255

                if not draw_all:
                    img[frame_num, rr, cc, 1:3] = 255

        if save_file is not None:
            tifffile.imsave(save_file, data=img.astype('uint8'))
        else:
            return img.astype('uint8')

    def work_with_fire(self):
        def calc_distance_track_filament(track, filament):
            distance_from_fil_to_track = []
            for coord in filament.coords:
                distance_from_fil_to_track.append(min([np.sqrt(((point - coord) ** 2).sum()) for point in track]))
            sorted(distance_from_fil_to_track)
            return distance_from_fil_to_track[len(distance_from_fil_to_track) // 2]

        def assign_fire_track_for_each_filament(frame, fire_tracks, frame_num, min_dist_constant=5):
            # before doing this clean fire_tracks
            for ft in fire_tracks:
                ft.frames_filaments[frame_num] = []

            for filament in frame.filaments:
                min_dist = 10_000
                min_ind = -1
                for ind, ft in enumerate(fire_tracks):

                    max_x, max_y = np.max(np.array(ft.branch.points), axis=0)
                    min_x, min_y = np.min(np.array(ft.branch.points), axis=0)

                    if min_x - min_dist_constant <= filament.center_x <= max_x + min_dist_constant \
                            and min_y - min_dist_constant <= filament.center_y <= max_y + min_dist_constant:
                        ### look for closest filaments, not very distant
                        dist = calc_distance_track_filament(ft.branch.points, filament)
                        if min_dist > dist:
                            min_dist = dist
                            min_ind = ind
                if min_dist < min_dist_constant:
                    fire_tracks[min_ind].add_filament(filament, min_dist, frame_num)

        def check_filament_is_on_end_of_trace(filament, fire_points):
            filament_start = filament.coords[0]
            filament_end = filament.coords[-1]
            s_point = find_closest_ind_in_array_to_point(fire_points, filament_start)
            e_point = find_closest_ind_in_array_to_point(fire_points, filament_end)
            print(s_point, e_point, len(fire_points))
            return abs(s_point) < 3 or abs(s_point - len(fire_points)) < 3 or \
                   abs(e_point) < 3 or abs(e_point - len(fire_points)) < 3

        def adjust_mean(mean_filament_length, count, new_length):
            mean_filament_length = ((mean_filament_length * count) + new_length) / (
                    count + 1)
            count += 1
            return mean_filament_length, count

        def check_right_trace_orientation(fire_track: FireTrack, check_frame_num: int):
            """
            Assume that we have 1 or -1 filament assigned (as it is called only from such places).
            check_frame_num tells us it is 1 or -1 frames.
            """
            def get_dists_0_m1(v, fire_track):
                fil: Filament = v[0][1]
                dist_to_0 = np.min(np.sum(np.sqrt((fil.coords - fire_track.branch.points[0]) ** 2), axis=1))
                dist_to_m1 = np.min(np.sum(np.sqrt((fil.coords - fire_track.branch.points[-1]) ** 2), axis=1))
                return dist_to_0, dist_to_m1

            for ind, (k, v) in enumerate(fire_track.frames_filaments.items()):
                if ind == 1 and k == check_frame_num:
                    dist_to_0, dist_to_m1 = get_dists_0_m1(v, fire_track)
                    if dist_to_m1 < dist_to_0:
                        fire_track.branch.points = fire_track.branch.points[::-1]
                elif ind == len(fire_track.frames_filaments) - 2 and k == check_frame_num:
                    dist_to_0, dist_to_m1 = get_dists_0_m1(v, fire_track)
                    if dist_to_m1 > dist_to_0:
                        fire_track.branch.points = fire_track.branch.points[::-1]

        def process_fire_cluster(video, fire_clusters, frame_num_start):
            all_changed_filaments = defaultdict(list)

            print(fire_clusters[0].branches)

            fire_tracks = []
            all_branches = []
            for ind, cl in enumerate(fire_clusters):
                for branch in cl.branches:
                    all_branches.append(branch)
                    fire_tracks.append(FireTrack(cl, ind, branch))

            from fire import visualize_branches

            #if to_display:
            # visualize_branches((512, 512), all_branches)
            for _ in range(1):
                ### TODO: change min_dist to some universal const
                for frame in list(video.frames)[
                             frame_num_start:frame_num_start + FIRE_NUM]:  # working with first fire_frame
                    frame_num = frame.num
                    assign_fire_track_for_each_filament(frame, fire_tracks, frame_num)

                # process one specific case!
                all_merged_fils = []
                frame2new_fils = defaultdict(list)
                for ft_num, ft in enumerate(fire_tracks):
                    frame_num2number_of_fils = {k: len(v) for k, v in ft.frames_filaments.items()}

                    print(ft, ft_num)
                    print(frame_num2number_of_fils)

                    def merge_filaments_that_were_broken():
                        pass

                    def split_filaments_that_were_merged():
                        pass

                    empty_frames = [k for k, v in frame_num2number_of_fils.items() if v == 0]

                    one_filament_frames = [k for k, v in frame_num2number_of_fils.items() if v == 1]
                    multiple_filaments_frames = [k for k, v in frame_num2number_of_fils.items() if v > 1]

                    if len(one_filament_frames) + len(multiple_filaments_frames) < 3:
                        continue

                    sum_length = 0
                    count = 0
                    for k, v in ft.frames_filaments.items():
                        if len(v) == 1:
                            sum_length += v[0][1].length
                            count += 1

                    if count == 0:
                        continue

                    mean_filament_length = sum_length / count

                    for ind, (k, v) in enumerate(ft.frames_filaments.items()):
                        if k in multiple_filaments_frames:
                            print('our case!')
                            new_length = 0
                            for dist, fil in v:
                                new_length += fil.length
                            # TODO: fix 10, 20 is very strange constant
                            # TODO: can be several different filaments on one fire track
                            mean_len = mean_filament_length
                            if abs(new_length - mean_len) < 20:
                                print('really our case!')
                                ### check that filaments are really one
                                ### not two different filaments on one track
                                ### how to do that?

                                fils_to_merge = [fil for _, fil in v]

                                ### TODO: fix assume that we have only two fils that need to be merged
                                # need to connect two filaments the right way
                                if len(fils_to_merge) != 2:
                                    continue

                                first_filament = fils_to_merge[0]
                                second_filament = fils_to_merge[1]

                                new_filament = connect_two_filaments(first_filament, second_filament)

                                if new_filament is None:
                                    # TODO: what filament is the right one?
                                    # The one that is more similar wrt to lengths
                                    continue

                                frame2new_fils[k].append(new_filament)
                                ft.frames_filaments[k] = [[1.0, new_filament]]

                                all_merged_fils.extend(fils_to_merge)

                                all_changed_filaments[k].extend(fils_to_merge)

                                one_filament_frames.append(k)

                                mean_filament_length, count = adjust_mean(mean_filament_length, count, new_filament.length)

                    print(empty_frames)
                    for k in empty_frames:
                        if k == frame_num_start + FIRE_NUM - 1 and k - 1 in one_filament_frames:
                            print('it is a last one!')
                            future_filament = None
                            prev_filament = ft.frames_filaments[k - 1][0][1]
                            if check_filament_is_on_end_of_trace(prev_filament, ft.branch.points):
                                continue
                            check_right_trace_orientation(ft, k - 1)
                        elif k == frame_num_start and k + 1 in one_filament_frames:
                            print('it is a first one')
                            future_filament = ft.frames_filaments[k + 1][0][1]
                            prev_filament = None
                            if check_filament_is_on_end_of_trace(future_filament, ft.branch.points):
                                continue
                            check_right_trace_orientation(ft, k + 1)
                        elif k - 1 in one_filament_frames and k + 1 in one_filament_frames:
                            ### can interpolate???, try to reconst k
                            future_filament = ft.frames_filaments[k + 1][0][1]
                            prev_filament = ft.frames_filaments[k - 1][0][1]
                        else:
                            continue

                        new_filament = reconstruct_filament(prev_filament, future_filament, mean_filament_length, ft)

                        ### TODO: dictionary change size during iteration !!!
                        # ft.frames_filaments[k-1].append([1.0, new_filament])
                        frame2new_fils[k].append(new_filament)
                        one_filament_frames.append(k)

                        all_changed_filaments[k].append(new_filament)

                        mean_filament_length, count = adjust_mean(mean_filament_length, count, new_filament.length)

                        distances_to_filaments = [calc_distance_track_filament(filament.coords.tolist(), new_filament) for
                                                  filament in video.frames[k].filaments]
                        min_dist = min(distances_to_filaments)

                        # TODO: fix random constant
                        # TODO: what if several min dist!!!
                        if min_dist < 3:
                            """
                            Try to find filament that included this new filament and cut it. 
                            """
                            min_ind = distances_to_filaments.index(min_dist)
                            too_long_filament = video.frames[k].filaments[min_ind]
                            relation = [
                                [min([np.sqrt(((coord - coord2) ** 2).sum()) for coord2 in new_filament.coords]), coord] for
                                coord in too_long_filament.coords]
                            print(relation)
                            shorted_coords = [coord for coord in too_long_filament.coords if min(
                                [np.sqrt(((coord - coord2) ** 2).sum()) for coord2 in new_filament.coords]) > 2]
                            # TODO: what if it breaks three long branches and as result only one will remain, but must remain two!
                            print(len(shorted_coords))
                            shorted_coords = get_biggest_chunk_array_2(shorted_coords)
                            if shorted_coords is not None:
                                print(len(shorted_coords))
                                new_shorted_filament = Filament(shorted_coords)
                                video.frames[k].filaments[min_ind] = new_shorted_filament

                                all_changed_filaments[k].append(new_shorted_filament)

                        print(distances_to_filaments)
                        print('empty frame')

                    ### change all firetracks from one cluster
                    fire_track_from_one_cluster = [f_track for f_track in fire_tracks if
                                                   f_track.cluster_num == ft.cluster_num]
                    for frame in list(video.frames)[
                                 frame_num_start:frame_num_start + FIRE_NUM]:  # working with first fire_frame
                        frame_num = frame.num
                        frame.filaments = [fil for fil in frame.filaments if fil not in all_merged_fils]
                        frame.filaments.extend(frame2new_fils[frame_num])
                        frame2new_fils[frame_num] = []  # all merged fils think about
                        assign_fire_track_for_each_filament(frame, fire_track_from_one_cluster, frame_num)

                for frame in list(video.frames)[
                             frame_num_start:frame_num_start + FIRE_NUM]:  # working with first fire_frames
                    frame.filaments = [fil for fil in frame.filaments if fil not in all_merged_fils]
                    frame.filaments.extend(frame2new_fils[frame.num])

            self.visualize_by_frames_2(all_changed_filaments, f'./{frame_num}_fire.tif')
            return fire_tracks

        if self.loaded_fire:
            return

        fire_tracks_list = []
        for ind, fire_clusters in enumerate(self.fire_frames):
            fire_tracks = process_fire_cluster(self, fire_clusters, ind * FIRE_NUM)
            fire_tracks_list.append(fire_tracks)

        MIN_FIL_LEN_AF = 6
        for frame in self.frames:
            frame.filaments = [filament for filament in frame.filaments if len(filament.coords) > 6]

        with open(self.path_to_results / 'save_fire.pkl', 'wb') as f:
            pkl.dump(self, f)

    def create_links_gnn(self, distance_type='cm',
                         gate_type='constant',
                         path_filament_len_type='last',
                         process_length_small_filaments_type='simple', len_small_filaments=20,
                         ratio_big_filaments=1.5, ratio_small_filaments=3.5,
                         add_filaments_fast=True):
        distance = choose_distance(distance_type)
        gate_func = choose_gate_func(gate_type)
        add_filament = choose_add_fil_func(add_filaments_fast)

        for frame_num, frame in enumerate(self.frames):
            frame.filaments = [filament for filament in frame.filaments if filament.length >= 3]

        for frame_num, frame in enumerate(self.frames):
            if frame_num == 0:
                self.tracks = Tracks(frame)  # initialize all paths with filaments from first frame
                continue
            # get all previous links!
            used_filaments = [0 for _ in frame.filaments]
            for path_num, path in enumerate(self.tracks.paths):
                if path.is_finished:
                    continue
                path.predict_next_state()
                distance_to_fils = [distance(path.next_filament_predicted, filament) for filament in frame.filaments]

                if distance_to_fils:
                    distance_to_fils = gate_func(path, distance_to_fils)

                if not distance_to_fils or min(distance_to_fils) == INF_DISTANCE:
                    path.is_finished = True
                    continue

                not_found = True
                is_path_finished = False
                while not_found:
                    closest_filament = min(range(len(distance_to_fils)), key=lambda i: distance_to_fils[i])
                    if distance_to_fils[closest_filament] == INF_DISTANCE:
                        # if there are no filaments left
                        is_path_finished = True
                        break

                    found_filament = frame.filaments[closest_filament]
                    filament_len = get_path_fil_length(path, path_filament_len_type)

                    max_length = max(found_filament.length, filament_len)
                    min_length = min(found_filament.length, filament_len)

                    if check_filament_length_ratio(max_length, min_length, ratio_big_filaments,
                                                   ratio_small_filaments, len_small_filaments,
                                                   process_length_small_filaments_type) == INF_DISTANCE:
                        distance_to_fils[closest_filament] = INF_DISTANCE
                        continue

                    used_filaments[closest_filament] += 1
                    if used_filaments[closest_filament] != 1:
                        # already was here, bad filament
                        distance_to_fils[closest_filament] = INF_DISTANCE
                    else:
                        not_found = False

                if is_path_finished:
                    path.is_finished = is_path_finished
                    continue

                if not not_found:
                    add_filament(path, found_filament)

            new_filaments = [ind for ind, number_of_usages in enumerate(used_filaments) if number_of_usages == 0]
            self.tracks.paths += [Path(first_frame_num=frame_num, filament=frame.filaments[fil_num]) for fil_num in
                                  new_filaments]

        number_of_tracks_before = len(self.tracks.paths)
        self.tracks.paths = [path for path in self.tracks.paths if len(path.filament_path) >= 3]
        print(number_of_tracks_before, len(self.tracks.paths))

    def create_links_nn_la(self, distance_type='cm',
                         gate_type='constant',
                         path_filament_len_type='last',
                         process_length_small_filaments_type='simple', len_small_filaments=20,
                         ratio_big_filaments=1.5, ratio_small_filaments=3.5,
                         add_filaments_fast=True, prediction_type='trivial'):
        """
        with or withput Kalman Filter we will try to estimate
        next element with linear assignment problem!
        :return:
        """
        distance = choose_distance(distance_type)
        gate_func = choose_gate_func(gate_type)
        add_filament = choose_add_fil_func(add_filaments_fast)

        for frame_num, frame in enumerate(self.frames):
            frame.filaments = [filament for filament in frame.filaments if filament.length != 0]
            if frame_num == 0:
                self.tracks = Tracks(frame, prediction_type)  # initialize all paths with filaments from first frame
                continue

            if frame_num == 35:
                print(9)
            # get all previous links!
            dist_matrix = []
            # map from alll paths to not-finished
            pidnf2pid = {}
            path_ind_notfinished = 0
            for path_ind, path in enumerate(self.tracks.paths):
                if path.is_finished:
                    continue
                cx, cy = path.filament_path[-1].center_x,  path.filament_path[-1].center_y
                # if 400 <= cx <= 420 and 120 <= cy <= 142:
                #     print('found him' * 50)

                path.predict_next_state()
                pidnf2pid[path_ind_notfinished] = path_ind
                path_ind_notfinished += 1

                distance_to_fils = [INF_DISTANCE for _ in frame.filaments]

                for fil_ind, f_filament in enumerate(frame.filaments):
                    # if 410 <= f_filament.center_x <= 430 and 120 <= f_filament.center_y <= 150:
                    #     print('found him' * 50)
                    filament_len = get_path_fil_length(path, path_filament_len_type)

                    max_length = max(f_filament.length, filament_len)
                    min_length = min(f_filament.length, filament_len)

                    if check_filament_length_ratio(max_length, min_length, ratio_big_filaments,
                                                   ratio_small_filaments, len_small_filaments,
                                                   process_length_small_filaments_type) == INF_DISTANCE:
                        distance_to_fils[fil_ind] = INF_DISTANCE
                        continue

                    distance_to_fils[fil_ind] = distance(path.next_filament_predicted, f_filament)

                distance_to_fils = gate_func(path, distance_to_fils)
                dist_matrix.append(distance_to_fils)

            dist_matrix = np.array(dist_matrix)
            # row_ind -- filaments from prev frames
            # col_ind -- filaments from frames

            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            path_nums_to_be_finished = [
                path_num_to_finish for path_num_to_finish in pidnf2pid if path_num_to_finish not in row_ind
            ]
            for path_num_to_finish in path_nums_to_be_finished:
                self.tracks.paths[pidnf2pid[path_num_to_finish]].is_finished = True
            used_filaments = [0 for _ in frame.filaments]
            for r_i, c_i in zip(row_ind, col_ind):
                # a_r_i -- for self.tracks.paths
                # r_i -- for cost_matrix
                a_r_i = pidnf2pid[r_i]
                path_to_continue = self.tracks.paths[a_r_i]
                if dist_matrix[r_i, c_i] == INF_DISTANCE:
                    path_to_continue.is_finished = True
                else:
                    used_filaments[c_i] += 1
                    filament_to_continue_path = frame.filaments[c_i]
                    add_filament(path_to_continue, filament_to_continue_path)

            new_filaments = [ind for ind, number_of_usages in enumerate(used_filaments) if number_of_usages == 0]
            self.tracks.paths += [Path(first_frame_num=frame_num, filament=frame.filaments[fil_num],
                                       prediction_type=prediction_type) for fil_num in new_filaments]

        print(len(self.tracks.paths))
        self.tracks.paths = [path for path in self.tracks.paths if len(path.filament_path) >= 3]
        print(len(self.tracks.paths))


def load_vid(path_to_results: pathlib.Path, use_fire: bool, forse_calc_fire=False) -> Video:
    # TODO: make better save
    loaded_fire = False

    if use_fire and not forse_calc_fire:
        path_to_fire = path_to_results / 'save_fire.pkl'
        if path_to_fire.exists():
            path_to_save = path_to_fire
            loaded_fire = True
        else:
            path_to_save = str(path_to_results / 'save.pkl')
    else:
        path_to_save = str(path_to_results / 'save.pkl')

    with open(path_to_save, 'rb') as f:
        vid = pkl.load(f)

    vid.loaded_fire = loaded_fire
    return vid


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_argparse():
    parser = argparse.ArgumentParser(description='Track processed tiff file with filaments')
    parser.add_argument('file_path', type=str, help="path to tiff file for tracking")
    parser.add_argument('--use_fire', type=str2bool, default=False, help='use fire for tracking')
    parser.add_argument('--dist_type', type=str, default='cm', choices=['cm', 'fast_like', 'cm_length'],
                        help='what type of distance to use between filaments')
    parser.add_argument('--mdf_to_save', type=str, default='gnn_cm.mdf',
                        help="name of saved mdf file with tracking results")
    parser.add_argument('--tracker_type', type=str, default='gnn', choices=['gnn', 'lap', 'kalman_lap'],
                        help="data association method to use")
    parser.add_argument('--prediction_type', type=str, default='trivial', choices=['trivial', 'kalman'],
                        help="predict next filament cm with kalman filter or not")
    parser.add_argument('--gate_type', type=str, default='constant',
                        choices=['constant', 'motion', 'motion_kalman', 'motion_dense'],
                        help="what type of gate to use for gnn method")
    parser.add_argument('--path_filament_len_type', type=str, default='last', choices=['last', 'mean'],
                        help="how to find len of filament in path")
    parser.add_argument('--process_length_small_filaments_type', type=str, default='simple',
                        choices=['simple', 'processing', 'no_processing'], help="process_small_filaments or not")
    parser.add_argument('--len_small_filaments', type=int, default=20, help="what filament is considered small")
    parser.add_argument('--ratio_big_filaments', type=float, default=1.5, help="ratio of lens for big fils")
    parser.add_argument('--ratio_small_filaments', type=float, default=3.5, help="ratio of lens for small fils")
    parser.add_argument('--add_filaments_fast', type=str2bool, default=True,
                        help="add new fils to track with fast overlap score or always using our method")
    parser.add_argument('--force_calculate_fire', type=str2bool, default=False,
                        help='whether to use saved fire result or not')
    return parser


def main(args):
    path_to_file = pathlib.Path(args.file_path)
    path_to_results = pathlib.Path('./results')
    path_to_results = path_to_results / path_to_file.stem

    vid = load_vid(path_to_results, args.use_fire, args.force_calculate_fire)
    vid.update_path_to_results(path_to_results)
    print(vid.path_to_results)
    # for ind, frame in enumerate(vid.frames):
    #     if ind >= 5 and ind <= 10:
    #         Filament.draw_filaments(frame.filaments, np.zeros([512, 512, 3]),
    #                                 path_to_save=f'/Users/danilkononykhin/Desktop/{ind}.png')

    # vid.visualize_by_frames(f'/Users/danilkononykhin/Desktop/res_{path_to_file.stem}.tif')

    if args.use_fire:
        vid.work_with_fire()

    # vid.visualize_by_frames(f'/Users/danilkononykhin/Desktop/res_{path_to_file.stem}_fire.tif')
    # for ind, frame in enumerate(vid.frames):
    #     if ind >= 5 and ind <= 10:
    #         Filament.draw_filaments(frame.filaments, np.zeros([512, 512, 3]),
    #                                 path_to_save=f'/Users/danilkononykhin/Desktop/{ind}_fire.png')

    if args.tracker_type == 'gnn':
        vid.create_links_gnn(
            distance_type=args.dist_type,
            gate_type=args.gate_type,
            path_filament_len_type=args.path_filament_len_type,
            process_length_small_filaments_type=args.process_length_small_filaments_type,
            len_small_filaments=args.len_small_filaments,
            ratio_big_filaments=args.ratio_big_filaments,
            ratio_small_filaments=args.ratio_small_filaments,
            add_filaments_fast=args.add_filaments_fast
        )
    elif args.tracker_type == 'lap':
        vid.create_links_nn_la(
            distance_type=args.dist_type,
            gate_type=args.gate_type,
            path_filament_len_type=args.path_filament_len_type,
            process_length_small_filaments_type=args.process_length_small_filaments_type,
            len_small_filaments=args.len_small_filaments,
            ratio_big_filaments=args.ratio_big_filaments,
            ratio_small_filaments=args.ratio_small_filaments,
            add_filaments_fast=args.add_filaments_fast,
            prediction_type=args.prediction_type
        )
    else:
        print('Something is wrong. Not known tracker type')
        return
    vid.tracks._save_data_to_mtrackj_format_new(path_to_results / args.mdf_to_save)

    pathlib.Path('./results_folders').mkdir(exist_ok=True)
    path_to_folder_results = pathlib.Path('./results_folders') / args.mdf_to_save
    path_to_folder_results.mkdir(exist_ok=True)
    vid.tracks._save_data_to_mtrackj_format_new(path_to_folder_results / path_to_file.with_suffix('.mdf').name)

    vid.visualize_by_frames_gates(path_to_file, save_file='/Users/danilkononykhin/Desktop/first_vis_speed.tif')

    vid.visualize_by_frames_gates(
        path_to_file,
        save_file='/Users/danilkononykhin/Desktop/first_vis_coords.tif',
        gate_type='kalman_coords_est_square'
    )

    vid.visualize_by_frames_gates(
        path_to_file,
        save_file='/Users/danilkononykhin/Desktop/first_vis_ellipse.tif',
        gate_type='kalman_ellipsis_est_square'
    )

    vid.visualize_by_frames_gates(
        path_to_file,
        save_file='/Users/danilkononykhin/Desktop/first_vis_all.tif',
        gate_type='kalman_ellipsis_est_square',
        draw_all=True
    )

    vid.visualize_by_frames(save_file='/Users/danilkononykhin/Desktop/all_fils_viz.tif')

    # vid.visualize_by_frames(str(path_to_results / './what_filament_left_fire_2.tif'))
    # for frame in vid.frames:
    #     Filament.draw_filaments(frame.filaments, np.zeros([512, 512, 3]))


if __name__ == "__main__":
    parser = create_argparse()
    pargs = parser.parse_args()

    main(pargs)
