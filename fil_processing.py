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

FIRE_NUM = 5


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
    def __init__(self, frame=None):
        self.tips = []
        if frame is not None:
            self.paths = [Path(frame.num, filament) for filament in frame.filaments]
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
        img = np.zeros(shape=(1100, 1100))
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
    def draw_filaments(filaments, img=None, to_draw=True):
        if not isinstance(filaments, list):
            filaments = [filaments]
        if img is None:
            img = np.zeros(shape=(512, 512, 3))
        for filament in filaments:
            g_color = np.random.randint(1, 256)
            b_color = np.random.randint(1, 256)
            img[filament.xs, filament.ys, 1] = g_color
            img[filament.xs, filament.ys, 2] = b_color
            print(g_color, b_color)
            img[filament.center_x, filament.center_y, 0] = 255

        if to_draw:
            plt.imshow(img, interpolation='bicubic')
            plt.show()
        else:
            return img


class Path:
    def __init__(self, first_frame_num, filament):
        self.filament_path = [filament]
        self.first_frame_num = first_frame_num
        self.is_finished = False
        self.next_center_velocity = np.array([[0, 0]])

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
        # if len(self.filament_path) >= 2:
        #     print(1)
        return self.filament_path[-1]
        # new_filament_coords = self.filament_path[-1].coords + self.next_center_velocity
        # return Filament(new_filament_coords.tolist(),
        #                 self.filament_path[-1].number_of_tips)

    def predict_next_state(self):
        # if len(self.filament_path) < 2:
        #     self.next_center_velocity = np.array([[0, 0]])
        #     return
        # prev_center = self.filament_path[-2].cm
        # curr_center = self.filament_path[-1].cm
        # self.next_center_velocity = curr_center - prev_center
        pass

    def add(self, filament):
        self.filament_path.append(filament)


def get_biggest_chunk_array(fil):
    branches = np.where(np.abs(np.diff(np.array(fil), axis=0)) > 1)[0]
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


# TODO: change this later as was discussed!!!
def process_filaments_simple(filament_list):
    # all_branches = []
    filaments = []
    for fil in filament_list:
        fil = get_biggest_chunk_array(fil)
        if fil is None:
            continue

        filament = Filament(fil)
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


def choose_distance(type_):
    if type_ == 'cm':
        return Filament.fils_distance_cm
    elif type_ == 'fast_like':
        return Filament.fils_distance_fast


# from fire import Cluster
# from typing import List
from collections import defaultdict, namedtuple


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
                 processing_frame: str):
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

    def visualize_by_frames(self, save_file):
        img = np.zeros(shape=[len(self.frames), self.width, self.height, 3])
        for frame_num, frame in enumerate(self.frames):
            for _, filament in enumerate(frame.filaments):
                img[frame_num][filament.xs, filament.ys, 1:3] = 255
                img[frame_num][filament.center_x, filament.center_y, 0] = 255
        tifffile.imsave(save_file, data=img.astype('uint8'))

    def work_with_fire(self):
        def calc_distance_track_filament(track, filament):
            distance_from_fil_to_track = []
            for coord in filament.coords:
                distance_from_fil_to_track.append(min([np.sqrt(((point - coord) ** 2).sum()) for point in track]))
            sorted(distance_from_fil_to_track)
            return distance_from_fil_to_track[len(distance_from_fil_to_track) // 2]

        def assign_fire_track_for_each_filament(frame, fire_tracks, frame_num, min_dist_constant=5):
            for filament in frame.filaments:
                min_dist = 10_000
                min_ind = -1
                for ind, ft in enumerate(fire_tracks):
                    ### look for closest filaments, not very distant
                    dist = calc_distance_track_filament(ft.branch.points, filament)
                    if min_dist > dist:
                        min_dist = dist
                        min_ind = ind
                if min_dist < min_dist_constant:
                    fire_tracks[min_ind].add_filament(filament, min_dist, frame_num)

        def process_fire_cluster(video, fire_clusters, frame_num_start):
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

            ### TODO: change min_dist to some universal const
            for frame in list(video.frames)[
                         frame_num_start:frame_num_start + FIRE_NUM]:  # working with first fire_frame
                frame_num = frame.num
                assign_fire_track_for_each_filament(frame, fire_tracks, frame_num)

            # process one specific case!
            all_merged_fils = []
            frame2new_fils = defaultdict(list)
            for ft_num, ft in enumerate(fire_tracks):
                ### Filter too short tracks
                ### TODO: make constant!
                frame_num2number_of_fils = {k: len(v) for k, v in ft.frames_filaments.items()}
                if len(frame_num2number_of_fils) < 3:
                    continue

                print(ft, ft_num)
                print(frame_num2number_of_fils)

                def merge_filaments_that_were_broken():
                    pass

                def predict_missed_filament():
                    pass

                possible_frames = list(range(frame_num_start, frame_num_start + FIRE_NUM))
                existing_frames = [k for k, _ in ft.frames_filaments.items()]
                empty_frames = [el for el in possible_frames if el not in existing_frames]

                was_only_one = False
                lengths_filaments = []
                for ind, (k, v) in enumerate(ft.frames_filaments.items()):
                    if k == 6:
                        print(6)
                    if ind == 0 and len(v) == 1:
                        was_only_one = True
                        lengths_filaments.append(v[0][1].length)
                    elif ind == 0:
                        ### maybe we can try to repair filament track
                        ### that was like that: 2 1 1 1 1 1
                        ### can repair with information from the future
                        print('very bad!')
                        break
                    elif len(v) == 1:
                        was_only_one = True
                        lengths_filaments.append(v[0][1].length)
                    elif len(v) != 1 and was_only_one:
                        print('our case!')
                        new_length = 0
                        for dist, fil in v:
                            new_length += fil.length
                        # TODO: fix 10, 20 is very strange constant
                        # TODO: can be several different filaments on one fire track
                        mean_len = sum(lengths_filaments) / len(lengths_filaments)
                        if abs(new_length - mean_len) < 20:
                            print('really our case!')
                            ### check that filaments are really one
                            ### not two different filaments on one track
                            ### how to do that?

                            fils_to_merge = [fil for _, fil in v]

                            ### TODO: fix assume that we have only two fils that need to be merged
                            # need to connect two filaments the right way
                            assert len(fils_to_merge) == 2

                            first_filament = fils_to_merge[0]
                            second_filament = fils_to_merge[1]

                            new_filament = connect_two_filaments(first_filament, second_filament)

                            if new_filament is None:
                                # lengths_filaments.append(new_filament.length)
                                # TODO: what filament is the right one?
                                # The one that is more similar wrt to lengths
                                continue

                            frame2new_fils[k].append(new_filament)
                            ft.frames_filaments[k] = [[1.0, new_filament]]
                            was_only_one = True
                            all_merged_fils.extend(fils_to_merge)
                            # TODO: add to length some length here ? (done?)
                            lengths_filaments.append(new_filament.length)

                if len(lengths_filaments) == 0:
                    continue
                mean_filament_length = sum(lengths_filaments) / len(lengths_filaments)

                print(empty_frames)
                for k in empty_frames:
                    ### TODO: to think of something
                    ### TODO: check that we really can do this
                    if k == frame_num_start + FIRE_NUM - 1 and k - 1 in existing_frames:
                        print('it is a last one!')
                        filaments = ft.frames_filaments[k - 1]
                        if len(filaments) != 1:
                            continue
                        filament_start = filaments[0][1].coords[0]
                        filament_end = filaments[0][1].coords[-1]
                        fire_points = ft.branch.points
                        s_point = find_closest_ind_in_array_to_point(fire_points, filament_start)
                        e_point = find_closest_ind_in_array_to_point(fire_points, filament_end)
                        print(s_point, e_point, len(fire_points))
                        if abs(s_point) < 3 or abs(s_point - len(fire_points)) < 3 or \
                                abs(e_point) < 3 or abs(e_point - len(fire_points)) < 3:
                            continue
                        ### if filament is on the border of fire track, then we need to continue something

                        future_filament = None
                    elif k - 1 in existing_frames and k + 1 in existing_frames:
                        ### can interpolate???, try to reconst k
                        future_filament = ft.frames_filaments[k + 1][0][1]
                    else:
                        continue

                    prev_filament = ft.frames_filaments[k - 1][0][1]
                    new_filament = reconstruct_filament(prev_filament, future_filament, mean_filament_length, ft)

                    ### TODO: dictionary change size during iteration !!!
                    # ft.frames_filaments[k-1].append([1.0, new_filament])
                    frame2new_fils[k].append(new_filament)

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
                    frame2new_fils[frame_num] = []
                    assign_fire_track_for_each_filament(frame, fire_track_from_one_cluster, frame_num)

            for frame in list(video.frames)[
                         frame_num_start:frame_num_start + FIRE_NUM]:  # working with first fire_frames
                frame.filaments = [fil for fil in frame.filaments if fil not in all_merged_fils]
                frame.filaments.extend(frame2new_fils[frame.num])

            return fire_tracks

        fire_tracks_list = []
        for ind, fire_clusters in enumerate(self.fire_frames):
            fire_tracks = process_fire_cluster(self, fire_clusters, ind * FIRE_NUM)
            fire_tracks_list.append(fire_tracks)

    def create_links_gnn(self, distance_type='cm'):
        distance = choose_distance(distance_type)
        for frame_num, frame in enumerate(self.frames):
            frame.filaments = [filament for filament in frame.filaments if filament.length >= 3]
        for frame_num, frame in enumerate(self.frames):
            print(frame_num)
            if frame_num == 3:
                print(3)
            print(len(frame.filaments))
            # frame.filaments = [el for el in frame.filaments if el.number_of_tips == 2]
            if frame_num == 0:
                self.tracks = Tracks(frame)  # initialize all paths with filaments from first frame
                continue
            # get all previous links!
            used_filaments = [0 for _ in frame.filaments]
            for path_num, path in enumerate(self.tracks.paths):
                if path_num + 1 == 5:
                    print(5)
                if np.abs(path.filament_path[-1].center_x - 358) < 10 and np.abs(
                        path.filament_path[-1].center_y - 236) < 10:
                    print('ha')
                if path.is_finished:
                    continue
                path.predict_next_state()
                distance_to_fils = [distance(path.next_filament_predicted, filament) for filament in frame.filaments]

                not_found = True
                is_path_finished = False

                while not_found:
                    if not distance_to_fils or min(distance_to_fils) > 39:  # when to stop tracking!
                        is_path_finished = True
                        not_found = False
                        break

                    closest_filament = min(range(len(distance_to_fils)), key=lambda i: distance_to_fils[i])
                    found_filament = frame.filaments[closest_filament]
                    max_length = max(found_filament.length, path.next_filament_predicted.length)
                    min_length = min(found_filament.length, path.next_filament_predicted.length)

                    if max_length / min_length > 1.5:
                        ### bad filament
                        distance_to_fils[closest_filament] = 10_000  # very big number
                        continue

                    used_filaments[closest_filament] += 1
                    if used_filaments[closest_filament] != 1:
                        # already was here, bad filament
                        distance_to_fils[closest_filament] = 10_000
                    else:
                        not_found = False

                if is_path_finished:
                    path.is_finished = is_path_finished
                    continue

                ##### GET OVERLAP
                prev_filament = path.next_filament_predicted
                if Filament.overlap_score_fast(prev_filament, found_filament) < 0:
                    found_filament = Filament(found_filament.coords[::-1],
                                              number_of_tips=found_filament.number_of_tips)
                #########
                path.add(found_filament)
                if len(path.filament_path) == 2:
                    second_filament = path.filament_path[-1]
                    first_filament = path.filament_path[-2]
                    head = second_filament.head
                    tail = second_filament.tail
                    min_head = min([np.sqrt(((point - head) ** 2).sum()) for point in first_filament.coords])
                    min_tail = min([np.sqrt(((point - tail) ** 2).sum()) for point in first_filament.coords])
                    if min_tail > min_head:
                        path.filament_path = [
                            Filament(first_filament.coords[::-1], number_of_tips=first_filament.number_of_tips),
                            Filament(second_filament.coords[::-1], number_of_tips=second_filament.number_of_tips)]

            new_filaments = [ind for ind, number_of_usages in enumerate(used_filaments) if number_of_usages == 0]
            self.tracks.paths += [Path(first_frame_num=frame_num, filament=frame.filaments[fil_num]) for fil_num in
                                  new_filaments]

        print(len(self.tracks.paths))
        self.tracks.paths = [path for path in self.tracks.paths if len(path.filament_path) >= 3]
        print(len(self.tracks.paths))

    def create_links_nn_la(self, distance_type='cm'):
        """
        witout Kalman Filter we will try to estimate
        next element with linear assignment problem!
        :return:
        """
        distance = choose_distance(distance_type)
        for frame_num, frame in enumerate(self.frames):
            # frame.filaments = [el for el in frame.filaments if el.number_of_tips() == 2]
            if frame_num == 0:
                self.tracks = Tracks(frame)  # initialize all paths with filaments from first frame
                continue
            # get all previous links!
            dist_matrix = []
            # map from alll paths to not-finished
            pidnf2pid = {}
            path_ind_notfinished = 0
            for path_ind, path in enumerate(self.tracks.paths):
                if path.is_finished:
                    continue
                path.predict_next_state()
                pidnf2pid[path_ind_notfinished] = path_ind
                path_ind_notfinished += 1
                distance_to_fils = [distance(path.next_filament_predicted, filament) for filament in
                                    frame.filaments]
                dist_matrix.append(distance_to_fils)

            dist_matrix = np.array(dist_matrix)
            # row_ind -- filaments from prev frames
            # col_ind -- filaments from frames
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            used_filaments = [0 for _ in frame.filaments]
            for r_i, c_i in zip(row_ind, col_ind):
                # a_r_i -- for self.tracks.paths
                # r_i -- for cost_matrix
                a_r_i = pidnf2pid[r_i]
                path_to_continue = self.tracks.paths[a_r_i]
                if dist_matrix[r_i, c_i] > path_to_continue.last_filament.length // 2:
                    path_to_continue.is_finished = True
                else:
                    filament_to_continue_path = frame.filaments[c_i]
                    path_to_continue.add(filament_to_continue_path)
                    used_filaments[c_i] += 1
            new_filaments = [ind for ind, number_of_usages in enumerate(used_filaments) if number_of_usages == 0]
            self.tracks.paths += [Path(first_frame_num=frame_num, filament=frame.filaments[fil_num]) for fil_num in
                                  new_filaments]

        print(len(self.tracks.paths))
        self.tracks.paths = [path for path in self.tracks.paths if len(path.filament_path) >= 2]
        print(len(self.tracks.paths))


def load_vid(path_to_results: pathlib.Path) -> Video:
    # TODO: make better save
    path_to_save = path_to_results / 'save.pkl'
    with open(path_to_save, 'rb') as f:
        vid = pkl.load(f)
    return vid


def create_argparse():
    parser = argparse.ArgumentParser(description='Track processed tiff file with filaments')
    parser.add_argument('file_path', type=str, help="path to tiff file for tracking")
    parser.add_argument('--use_fire', type=bool, default=False, help='use fire for tracking')
    parser.add_argument('--dist_type', type=str, default='cm', choices=['cm', 'fast_like'],
                        help='what type of distance to use between filaments')
    parser.add_argument('--mdf_to_save', type=str, default='gnn_cm.mdf',
                        help="name of saved mdf file with tracking results")
    return parser


def main(args):
    path_to_file = pathlib.Path(args.file_path)
    path_to_results = pathlib.Path('./results')
    path_to_results = path_to_results / path_to_file.stem

    vid = load_vid(path_to_results)

    # for frame in vid.frames:
    #     Filament.draw_filaments(frame.filaments, np.zeros([512, 512, 3]))
    if args.use_fire:
        vid.work_with_fire()

    vid.create_links_gnn(distance_type=args.dist_type)
    vid.tracks._save_data_to_mtrackj_format_new(path_to_results / args.mdf_to_save)
    # vid.visualize_by_frames('./visualized_pickle.tif')
    # vid.visualize_by_frames(str(path_to_results / './what_filament_left_fire_2.tif'))
    # for frame in vid.frames:
    #     Filament.draw_filaments(frame.filaments, np.zeros([512, 512, 3]))


if __name__ == "__main__":
    parser = create_argparse()
    pargs = parser.parse_args()

    main(pargs)
