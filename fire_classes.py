

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

    # TODO: fix self.first_nucl_point in branches after merging!
    def __repr__(self):
        return str(self.first_nucl_point) + ' : ' + str(self.points) + '\n'

    def __len__(self):
        return len(self.points)