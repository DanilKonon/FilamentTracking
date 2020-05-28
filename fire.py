### FIRE ALGORITHM IMPLEMENTATION
from __future__ import annotations

from skimage.external import tifffile
import numpy as np
from matplotlib import pyplot as plt
from main import norm_image, create_steer_filter_1, my_canny
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from math import ceil
from copy import deepcopy
from motmetrics.distances import norm2squared_matrix
import cv2
from skimage.filters import gaussian
from scipy.ndimage.morphology import distance_transform_edt
from collections import namedtuple
from skimage.filters import median
from typing import Tuple, List
import queue

# TODO: fix all consts!
NEAR_RADIUS = 10


class NucleationPoint:
    def __init__(self, point):
        self.point = point
        self.i, self.j = point
        self.branches = []

    def start_branches(self, transformaed_bi, to_display=False):
        """
        Returns how much branches were created for this nucl point
        """
        i, j = self.i, self.j
        radius_to_look = ceil(transformaed_bi[self.i, self.j])
        box_to_look = get_box(transformaed_bi,
                              self.i, self.j,
                              radius_to_look)
        if not check_coords(transformaed_bi,
                            self.i, self.j,
                            radius_to_look):
            return 0

        if to_display:
            print(transformaed_bi[i, j], radius_to_look)
            print(box_to_look)

        peaks = find_local_maxima_on_border(box_to_look)
        peaks = [(val, i, j) for val, i, j in peaks if val > THETA_LMP]
        ### TODO: add theta_lmpdist!!!
        peaks = [(val, *get_real_coordinates(i, j, i_1, j_1, radius_to_look)) for val, i_1, j_1 in peaks]
        peaks = [(val, i_1, j_1) for val, i_1, j_1 in peaks if check_coords(transformaed_bi, i_1, j_1, radius_to_look)]

        if to_display:
            print(f'peaks: {peaks}')

        for val, i_1, j_1 in peaks:
            if to_display:
                print('*' * 250)
                print('begin new branch: ', val, i, j, i_1, j_1)
                print(box_to_look)

            branch = Branch(
                self,
                np.array([i_1, j_1])
            )

            self.branches.append(branch)

        return len(peaks)

    def __repr__(self):
        return f'{self.i}, {self.j}: {len(self.branches)}'

    def __eq__(self, other):
        if isinstance(other, tuple):
            return (self.i == other[0]) and (self.j == other[1])
        elif isinstance(other, NucleationPoint):
            return (self.i == other.i) and (self.j == other.j)

    def __hash__(self):
        return f'{self.i}_{self.j}'.__hash__()


class Branch:
    eps = 0.000001

    def __init__(self,
                 first_point: NucleationPoint,
                 second_point):
        """

        :param first_point: first nucl point
        :param second_point: np.array(tuple(int, int))
        """
        first_point_arr = np.array(first_point.point)
        self.points = [first_point_arr, second_point]
        self.dir = calc_direction(first_point_arr, second_point)
        self.dirs = [self.dir]
        self.first_nucl_point = first_point  # TODO: SHOULD I DO THIS NUCLEATION POINT CLASS??
        self.last_nucl_point = None

    def check_lmp(self, delta):
        cos_dist = np.sum(delta * self.dir)
        return cos_dist, cos_dist < np.cos(theta_ext)

    def update(self, point):
        if isinstance(point, NucleationPoint):
            point = np.array(point.point)
        if np.array_equal(self.points[-1], point):
            print('same point tried to be added')
            return
        delta = calc_direction(self.points[-1], point)
        self.dir = lambda_dirdecay * self.dir + (1 - lambda_dirdecay) * delta
        self.dir = self.dir / (np.sqrt(np.sum(self.dir ** 2)) + self.eps)
        print('new dir: ', self.dir)
        self.points.append(point)
        self.dirs.append(self.dir)

    def reverse(self):
        first_point = self.points[-1]
        second_point = self.points[-2]
        left_points = self.points[::-1][2:]
        self.points = [first_point, second_point]
        self.dir = calc_direction(first_point, second_point)
        self.dirs = [self.dir]
        self.first_nucl_point, self.last_nucl_point = self.last_nucl_point, self.first_nucl_point  # problem with reversed and first point always is nucl point
        self.update_points(left_points)  # ? [1, 2, 3, 4, 5, 6, 7][-3::-1]  ?
        return self

    def update_points(self, points):
        for point in points:
            self.update(point)

    def update_by_branch(self, branch: Branch):
        self.update_points(branch.points)
        self.last_nucl_point = branch.last_nucl_point

    # TODO: fix self.first_nucl_point in branches after merging!
    def __repr__(self):
        return str(self.first_nucl_point) + ' : ' + str(self.points) + '\n'

    def __len__(self):
        return len(self.points)


class Cluster:
    def __init__(self):
        self.nucl_points = []
        self.branches = []

    def add_branch(self, branch):
        self.branches.append(branch)

    def add_nucl_point(self, nucl_point):
        self.nucl_points.append(nucl_point)

    def __repr__(self):
        return 'branch: ' + repr(self.nucl_points) + '\n' + repr(self.branches)


def calc_min_distance_bt_four_points(start_point_o, end_point_o, start_point_d, end_point_d):
    """
    Calculates distance between each point from first object and each object from second object
    returns min_ind, min_dist
    """
    dist1 = calc_distance(start_point_o, start_point_d)  ## not use, it is always nucleous point!
    dist2 = calc_distance(start_point_o, end_point_d)
    dist3 = calc_distance(end_point_o, start_point_d)
    dist4 = calc_distance(end_point_o, end_point_d)
    dists = [dist1, dist2, dist3, dist4]
    ### TODO: assert that only one min is there
    min_ind, min_dist = min(enumerate(dists), key=lambda x: x[1])
    print(min_ind, min_dist)
    return min_ind, min_dist


def find_near_branch(branch_orig, branches, ind_to_skip):
    good_branches = []
    start_point_o = branch_orig.points[0]
    end_point_o = branch_orig.points[-1]
    for ind, branch_dupl in enumerate(branches):
        if ind == ind_to_skip:
            continue
        start_point_d = branch_dupl.points[0]
        end_point_d = branch_dupl.points[-1]

        min_ind, min_dist = calc_min_distance_bt_four_points(start_point_o, end_point_o,
                                                             start_point_d, end_point_d)

        if min_dist < NEAR_RADIUS:
            if min_ind == 0:
                good_branches.append((ind, 0, 0, min_dist))
            elif min_ind == 1:
                # index of branch, index to get from orig, index to get from dupl
                good_branches.append((ind, 0, -1, min_dist))
            elif min_ind == 2:
                good_branches.append((ind, -1, 0, min_dist))
            elif min_ind == 3:
                good_branches.append((ind, -1, -1, min_dist))
    return good_branches


def prune_branches(branches: List[Branch], to_display=False):
    branches_to_delete = []
    ind = 0
    ### is it good?
    ### get rid of all the not developed branhces ()
    # branches = [br for br in branches if len(br.points) > 3]
    while ind < len(branches):
        not_changed = True
        branch = branches[ind]
        good_branches = find_near_branch(branch, branches, ind_to_skip=ind)  # branches for continuation
        good_branches_filtered = []
        good_branches_f = []

        if to_display:
            print('before filtering: ', good_branches)

        for (b, or_ind, dupl_ind, dist) in good_branches:
            sim_val = abs(1 - abs(np.sum(branches[b].dirs[dupl_ind] * branch.dirs[or_ind])))
            is_similar_dirs = sim_val < 0.5
            #         if np.isnan(sim_val):
            #             ind = 1000
            #             break

            if to_display:
                print(b, or_ind, dupl_ind, dist, ' : ', sim_val)

            if is_similar_dirs:
                good_branches_filtered.append(b)
                good_branches_f.append((b, or_ind, dupl_ind, dist, sim_val))

        good_branches_start = [(b, or_ind, dupl_ind, dist, sim_val) for (b, or_ind, dupl_ind, dist, sim_val) in good_branches_f if
                               or_ind == 0]
        good_branches_end = [(b, or_ind, dupl_ind, dist, sim_val) for (b, or_ind, dupl_ind, dist, sim_val) in good_branches_f if
                             or_ind == -1]

        if to_display:
            print(good_branches_start)
            print(good_branches_end)
        #     print(good_branches_filtered, [np.sum(branches[b].dirs[dupl_ind] * branch.dirs[or_ind]) for (b, or_ind, dupl_ind, dist) in good_branches_f])

        def find_min_element(list_branches):
            # structure (b, or_ind, dupl_ind, dist, sim_val)
            # first choose with min distance
            # second choose with min sim_val
            branch_with_min_dist = min(list_branches, key=lambda x: x[-2])
            closest_branches = [x for x in list_branches if x[-2] == branch_with_min_dist[-2]]
            min_sim_val_branch = min(closest_branches, key=lambda x: x[-1])
            if len([x for x in closest_branches if x[-1] == min_sim_val_branch[-1]]) != 1:
                print('very bad')
            return min_sim_val_branch

        if len(good_branches_end) != 0:
            chosen_branch_idxs = find_min_element(good_branches_end)
            print('end', chosen_branch_idxs)
            chosen_branch_ind = chosen_branch_idxs[0]
            if chosen_branch_idxs is not None:
                chosen_branch = branches[chosen_branch_ind]
                if chosen_branch_idxs[2] == 0:
                    branch.update_by_branch(chosen_branch)
                else:  # dupl_ind == -1
                    chosen_branch.reverse()
                    branch.update_by_branch(chosen_branch)
                branches_to_delete.append(chosen_branch_ind)
                not_changed = False

        if len(good_branches_start) != 0:
            chosen_branch_idxs = find_min_element(good_branches_start)
            print('start', chosen_branch_idxs)
            chosen_branch_ind = chosen_branch_idxs[0]
            if chosen_branch_idxs is not None:
                dupl_branch = branches[chosen_branch_ind]
                if chosen_branch_idxs[2] == 0:
                    dupl_branch.reverse()
                    dupl_branch.update_by_branch(branch)
                else:  # dupl_ind == -1
                    dupl_branch.update_by_branch(branch)
                branches_to_delete.append(ind)
                not_changed = False

        print('changed', not_changed)

        if not not_changed:
            print(branches_to_delete)
            branches = [branch for ind_br, branch in enumerate(branches) if ind_br not in branches_to_delete]
            branches_to_delete = []
            ind = 0
        else:
            ind += 1

    return branches


def delete_excess_branches(branches):
    to_delete = []
    for ind1, branch in enumerate(branches):
        if branch.last_nucl_point is not None:
            for ind2, another_branch in enumerate(branches):
                if ind1 <= ind2 or another_branch.last_nucl_point is None:
                    continue
                if np.array_equal(branch.last_nucl_point, another_branch.first_nucl_point) and \
                        np.array_equal(another_branch.last_nucl_point, branch.first_nucl_point):
                    print(ind1, ind2)
                    to_delete.append(ind1)

    branches = [br for ind, br in enumerate(branches) if ind not in to_delete]
    return branches


def create_lighted_image(or_image, sigma, m_percent):
    image, theta = create_steer_filter_1(or_image, sigma, make_otsu_thresh=True)
    ridge_image = my_canny(image=image, theta=theta, m_percent=m_percent)

    ridge_image[ridge_image > 0] = 1
    #     ridge_image = skeletonize(ridge_image).astype('int')
    ridge_image = ridge_image.astype('int')
    ridge_image[ridge_image > 0] = 255

    lighted_image = np.zeros(shape=[ridge_image.shape[0], ridge_image.shape[1], 3])
    lighted_image[:, :, 0] = ridge_image
    lighted_image[:, :, 1] = or_image
    lighted_image[:, :, 2] = or_image

    return lighted_image


def find_nucleous_points(transformaed_bi):
    nucleous_points = []
    for i in range(s_box, transformaed_bi.shape[0] - s_box - 1):
        for j in range(s_box, transformaed_bi.shape[1] - s_box - 1):
            box = transformaed_bi[i - s_box:i + s_box + 1, j - s_box:j + s_box + 1]
            if np.sum(box) != 0 and np.sum(box == np.max(box)) == 1:
                index_of_max_elem = np.unravel_index(box.argmax(), box.shape)

                if index_of_max_elem == (s_box, s_box) and np.max(box) > theta_nuc:
                    real_index = i, j
                    nucleous_points.append(NucleationPoint(real_index))
                    nucleous_points[-1].start_branches(transformaed_bi)
    return nucleous_points


def find_local_maxima_on_border(box_to_look, theta_lmp=None):
    diameter_to_look = box_to_look.shape[0]
    peaks = []

    k = 0
    # print(box_to_look)
    prev = box_to_look[k, 0]
    for t in range(1, diameter_to_look - 1):
        next_ = box_to_look[k, t + 1]
        if box_to_look[k, t] >= prev and box_to_look[k, t] >= next_:
            peaks.append((box_to_look[k, t], k, t))
        prev = box_to_look[k, t]

    k = diameter_to_look - 1
    prev = box_to_look[k, 0]

    for t in range(1, diameter_to_look - 1):
        next_ = box_to_look[k, t + 1]
        if box_to_look[k, t] >= prev and box_to_look[k, t] >= next_:
            peaks.append((box_to_look[k, t], k, t))
        prev = box_to_look[k, t]

    t = 0
    prev = box_to_look[0, t]
    for k in range(1, diameter_to_look - 1):
        next_ = box_to_look[k + 1, t]
        if box_to_look[k, t] >= prev and box_to_look[k, t] >= next_:
            peaks.append((box_to_look[k, t], k, t))
        prev = box_to_look[k, t]

    t = diameter_to_look - 1
    prev = box_to_look[0, t]
    for k in range(1, diameter_to_look - 1):
        next_ = box_to_look[k + 1, t]
        if box_to_look[k, t] >= prev and box_to_look[k, t] >= next_:
            peaks.append((box_to_look[k, t], k, t))
        prev = box_to_look[k, t]

    prev = box_to_look[1, 0]
    next_ = box_to_look[0, 1]
    curr = box_to_look[0, 0]
    if curr >= prev and curr >= next_:
        peaks.append((curr, 0, 0))

    prev = box_to_look[0, diameter_to_look - 2]
    next_ = box_to_look[1, diameter_to_look - 1]
    curr = box_to_look[0, diameter_to_look - 1]
    if curr >= prev and curr >= next_:
        peaks.append((curr, 0, diameter_to_look - 1))

    prev = box_to_look[diameter_to_look - 2, 0]
    next_ = box_to_look[diameter_to_look - 1, 1]
    curr = box_to_look[diameter_to_look - 1, 0]
    if curr >= prev and curr >= next_:
        peaks.append((curr, diameter_to_look - 1, 0))

    prev = box_to_look[diameter_to_look - 1, diameter_to_look - 2]
    next_ = box_to_look[diameter_to_look - 2, diameter_to_look - 1]
    curr = box_to_look[diameter_to_look - 1, diameter_to_look - 1]
    if curr >= prev and curr >= next_:
        peaks.append((curr, diameter_to_look - 1, diameter_to_look - 1))

    if theta_lmp is not None:
        peaks = [(i, j, k) for (i, j, k) in peaks if i >= theta_lmp]

    return peaks


def get_box(image, i, j, s_box):
    if i in range(s_box, image.shape[0] - s_box - 1) and j in range(s_box, image.shape[1] - s_box - 1):
        return image[i - s_box:i + s_box + 1, j - s_box:j + s_box + 1]
    else:
        return None


def get_real_coordinates(i, j, i_1, j_1, s_box):
    return i + i_1 - s_box, j + j_1 - s_box


def check_coords(image, i, j, s_box):
    return s_box <= i <= image.shape[0] - s_box - 1 and s_box <= j <= image.shape[1] - 1 - s_box


def calc_distance(first_point, second_point):
    return np.sqrt(np.sum((first_point - second_point) ** 2))


EPS = 0.0000001


def calc_direction(first_point, second_point):
    return (second_point - first_point) / (np.sqrt(np.sum((second_point - first_point) ** 2)) + EPS)


def check_for_nucleous_points(i, j, s_box, nucleous_points, to_displ=False):
    xmin = i - s_box
    xmax = i + s_box
    ymin = j - s_box
    ymax = j + s_box

    if to_displ:
        print(f'checking for nucl point: {i}, {j}, {s_box}')
        print(f'xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}')

    res = []
    for nucl_point in nucleous_points:
        x, y = nucl_point.point
        if xmin <= x <= xmax and ymin <= y <= ymax:
            res.append(nucl_point)

    return res


def visualize_branches(shape_img: Tuple[int, int],
                       branches_to_draw: List[Branch], image_with_circles=None,
                       try_to_display_directions=False):
    """
    vis branches!
    :param shape_img:
    :param branches_to_draw:
    :param image_with_circles: use draw_nucl_points function to get image_with_circles!
    :return:
    """
    clean_image = np.zeros(list(shape_img) + [3])
    clean_image2 = clean_image

    for ind1, branch in enumerate(branches_to_draw):
        #     print(ind1)
        ind2 = ind1 + 1
        for point in branch.points:
            i, j = point
            #         clean_image2[i, j] = np.array([1/(num_branches+0.1)*ind2,
            #                                       1/(num_branches+0.1)*ind2,
            #                                       1/(num_branches+0.1)*ind2])
            clean_image2[i, j] = np.array([1, 1, 1])
    #         cv2.po(img=clean_image2,
    #                    center=(j, i), radius=1,
    #                    color=(255//(num_branches+120)*ind2,255//(num_branches+120)*ind2,255//(num_branches+120)*ind2),
    #                    thickness=-1)

    if image_with_circles is not None:
        clean_image2[:, :, 1] = image_with_circles / image_with_circles.max()

    plt.figure(figsize=(15, 15))

    if try_to_display_directions:
        clean_image2 *= 255
        for ind1, branch in enumerate(branches_to_draw):
            for p_num, p in enumerate(branch.points):
                if len(branch.points) - 1 == p_num:
                    continue
                dir_ = branch.dirs[p_num]
                i1, j1 = p
                i2, j2 = int(p[0]+ dir_[0]), int(p[1] + dir_[1])

                cv2.arrowedLine(clean_image2, (j1, i1), (j2+10, i2+10), (0, 0, 255), 1)
                # plt.arrow(i1, j1, dir_[0], dir_[1])
                # plt.text(j, i, f'{ind1}', color='red', fontsize=15)
        clean_image2 /= 255

    plt.imshow(clean_image2)
    for ind1, branch in enumerate(branches_to_draw):
        i, j = branch.points[0]
        plt.text(j, i, f'{ind1}', color='red', fontsize=15)

    plt.show()


def choose_right_branches(nucl_point, branch):
    branches_to_check = nucl_point.branches
    right_branches = []
    # we need to get rid of exactly one branch!
    closeness_to_sin180 = [abs(-1 - np.sum(branch.dirs[-1] * br.dirs[0])) for br in branches_to_check]
    most_close = min(closeness_to_sin180)
    for ind, el in enumerate(closeness_to_sin180):
        if el != most_close:
            right_branches.append(branches_to_check[ind])
        elif most_close > 0.2:
            right_branches.append(branches_to_check[ind])
    return right_branches


# need to add info about nucleous points
def extend_branch(dist_image, branch, nucleous_points, to_display=True) -> Tuple[bool, NucleationPoint]:
    """
    Returns tuple: (is_branch_finished_with_nucl_point, nucl_point)
    If is_branch_finished_with_nucl_point is False, then nucl_point -- NucleationPoint or None
    If we meet new nucleation point,
    Then if it has only two branches, one of them is incoming, then we can eliminate
    This nucleation point and merge two branches
    """
    i, j = branch.points[-1].tolist()

    if to_display:
        print('extending branch: ', i, j)

    radius_to_look = ceil(dist_image[i, j])
    box_to_look = get_box(dist_image, i, j, radius_to_look)

    if box_to_look is None:
        return False, None

    if to_display:
        print(box_to_look)

    possible_nucl_points = check_for_nucleous_points(i, j, radius_to_look, nucleous_points)

    if to_display:
        print(len(possible_nucl_points))
        print('nucl points', possible_nucl_points)

    if len(possible_nucl_points) != 0:
        if to_display:
            print(possible_nucl_points)
            print(branch.first_nucl_point)
        if branch.first_nucl_point != possible_nucl_points[0] and \
                (branch.last_nucl_point is None or branch.last_nucl_point != possible_nucl_points[0]):
            if to_display:
                print('found nucl point')
            end_nucl_point = possible_nucl_points[0]
            branch.update(end_nucl_point)
            branch.last_nucl_point = end_nucl_point
            #### finished branch with nucl point
            return True, end_nucl_point

    peaks = find_local_maxima_on_border(box_to_look, theta_lmp=THETA_LMP)
    peaks = [(val, *get_real_coordinates(i, j, i_1, j_1, radius_to_look)) for val, i_1, j_1 in peaks]
    peaks = [(val, i_1, j_1) for val, i_1, j_1 in peaks if check_coords(dist_image, i_1, j_1, radius_to_look)]

    print('peaks: ', peaks)
    if len(peaks) == 0:
        return False, None

    ### what if nucleous point is inside box!!!
    if to_display:
        print('begin looking for best_lmp')

    best_lmp = None
    curr_point = np.array([i, j])
    for val, i_1, j_1 in peaks:
        lmp_point = np.array([i_1, j_1])
        delta = calc_direction(curr_point, lmp_point)
        cos_dist, is_good = branch.check_lmp(delta)

        if to_display:
            print(cos_dist, lmp_point, is_good, np.arccos(cos_dist) < theta_ext)

        is_good = np.arccos(cos_dist) < theta_ext
        if not is_good:
            continue
        if best_lmp is None:
            best_lmp = (i_1, j_1, val)
        elif best_lmp[2] < val:
            best_lmp = (i_1, j_1, val)

    print('best_lmp: ', best_lmp)
    if best_lmp is None:
        # finished branch without nucl point
        return False, None

    branch.update(np.array(best_lmp[0:2]))
    return extend_branch(dist_image, branch, nucleous_points)


to_display = True
s_box = 5
theta_nuc = 1
THETA_LMP = 0.2
theta_ext = np.pi / 3
lambda_dirdecay = 0.5


def display_image(img):
    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(img)
    plt.show()


def process_pict_for_fire(picture, type_='max', to_display=False):
    """
    :param picture: shape of picture is [z, x, y]
    :param type_: more types are coming soon!
    :return: transformed image
    """
    if type_ == 'max':
        agg_picture = np.max(picture, axis=0)
    elif type_ == 'mean':
        agg_picture = np.mean(picture, axis=0)
    else:
        return None
    mean_picture = median(agg_picture, selem=np.ones(shape=[3, 3]))
    if to_display:
        plt.imshow(mean_picture, cmap='gray')
        plt.show()
    image, _ = create_steer_filter_1(mean_picture, sigma=2.0, make_otsu_thresh=False)
    image = median(image, selem=np.ones([5, 5]))
    thresh_otsu = threshold_otsu(image)
    binary_image = (image >= thresh_otsu).astype(int)

    if to_display:
        display_image(binary_image)

    transformaed_bi = distance_transform_edt(binary_image)
    transformaed_bi = gaussian(transformaed_bi)

    if to_display:
        display_image(transformaed_bi)

    return transformaed_bi


def load_image_example(
        file_path='/Users/danilkononykhin/PycharmProjects/Filaments/' + \
                  'Actin_filaments/Motility_Oct.19__tiff_mdf/Long_filaments_crossed_3.tif',
        first_frame=0,
        last_frame=5,
        type_='max',
        to_display=False
):
    t = tifffile.imread(str(file_path))
    t = t[first_frame:last_frame, :, :]
    t = t / 255
    return process_pict_for_fire(t, type_, to_display)


# transformaed_bi.min(), transformaed_bi.max()
# transformaed_bi = transformaed_bi[0:200, 340:450]

# ## TODO: make function out of this!
# plt.figure(figsize=(10, 10))
# plt.imshow(transformaed_bi)


def draw_nucl_points(image, nucleous_points):
    image_for_circles = norm_image(image.copy(), 255)

    for n_point in nucleous_points:
        i, j = n_point.point
        cv2.circle(img=image_for_circles, center=(j, i), radius=2, color=(255, 255, 255), thickness=-1)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_for_circles)
    plt.show()


def create_clusters(image, nucleous_points):
    clusters = []
    branches = []  # ???

    #### add branches queue!!!

    cluster = Cluster()
    already_visited = set()
    skipped_nucl_points = set()
    for nucl_point in nucleous_points:
        print(nucl_point)
        # if nucl_point.point[0] == 146:
            # print('aha')
        if nucl_point in already_visited:
            continue

        already_visited.add(nucl_point)
        cluster.add_nucl_point(nucl_point)
        branches_for_continuation = queue.LifoQueue()

        for br in nucl_point.branches:
            branches_for_continuation.put(br)

        while not branches_for_continuation.empty():
            branch = branches_for_continuation.get()
            is_nucl_point, n_point = extend_branch(image, branch, nucleous_points, True)
            print(n_point)
            if n_point is not None and n_point.point[0] == 146:
                print('found!')
            if n_point in already_visited:
                continue

            if not is_nucl_point:
                ### do something
                ### do nothing?
                # cluster.add_branch(branch)
                pass
            else:
                ### continue somehow
                cluster.add_nucl_point(n_point)
                # choose right branches wrt branch
                right_branches = choose_right_branches(n_point, branch)

                # if special case discussed above then:
                if len(right_branches) == 1 and len(n_point.branches) == 2:
                    branches_for_continuation.put(branch)
                    skipped_nucl_points.add(n_point)
                    already_visited.add(n_point)
                    continue
                    ### maybe delete n_point
                else:
                    # add this one to stack
                    for rb in right_branches:
                        branches_for_continuation.put(rb)
                already_visited.add(n_point)

            cluster.add_branch(branch)   ### check whether this branch changed everywhere
            branches.append(branch)

        clusters.append(cluster)
        cluster = Cluster()

    return clusters


def delete_short_branches(branches, min_length=3):
    return [br for br in branches if len(br) > min_length]


def prune_clusters(clusters):
    all_branches = []
    # TODO: work with close branches when in max mode especially
    for ind, cluster in enumerate(clusters):
        if ind == 14:
            print('ahahaha')
        print(cluster)
        br = prune_branches(cluster.branches)
        br = delete_short_branches(br, min_length=5)
        cluster.branches = br
        # if ind == 14:
        #     prune_branches(br)
        all_branches.extend(br)

    return clusters


def get_clusters_from_image(image=None):
    if image is None:
        image = load_image_example()
    nucleous_points_gl = find_nucleous_points(image)

    # if to_display:
    # draw_nucl_points(image, nucleous_points_gl)
    clusters = create_clusters(image, nucleous_points_gl)
    ### 4 кластер ! ##

    all_branches = []
    for ind, cl in enumerate(clusters):
        for branch in cl.branches:
            all_branches.append(branch)

    # if to_display:
    #     visualize_branches((512, 512), all_branches)

    clusters = prune_clusters(clusters)  # TODO: check do i need to save clusters!!!

    all_branches = []
    for ind, cl in enumerate(clusters):
        for branch in cl.branches:
            all_branches.append(branch)
    # if to_display:
    # visualize_branches((512, 512), all_branches)
    return clusters


# TODO: why zero clusters???
if __name__ == "__main__":
    file = '/Users/danilkononykhin/PycharmProjects/Filaments/' + \
            'Actin_filaments/Motility_Oct.19__tiff_mdf/Long_filaments_crossed_3.tif'

    fire_num = 5
    for i in range(3):
        if i != 2:
            continue

        clusters = get_clusters_from_image(
            load_image_example(file,
                               first_frame=i*fire_num,
                               last_frame=(i+1)*fire_num,
                               type_='max',
                               to_display=True)
        )
        for cluster in clusters:
            print(len(cluster.nucl_points), len(cluster.branches))

        all_branches = [br for cl in clusters for br in cl.branches]
        k = 0
        for ind, cl in enumerate(clusters):
            k += len(cl.branches)
            if k > 19:
                print(ind)
                break
        visualize_branches((512, 512), all_branches)