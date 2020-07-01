import argparse
import pickle as pkl
import shutil
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import figaspect
from numba import jit
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import convolve2d
from skimage import filters
from skimage.color import rgb2gray
from skimage.external import tifffile
# import tifffile
from skimage.morphology import skeletonize

from fil_processing import Video


# global g_x, g_y, g_xx, g_yy, g_xy


@jit(nopython=True)
def discretize_theta(im1_mag, theta):
    """
    Returns discretised theta
    """
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            el = theta[i][j]
            if im1_mag[i][j] <= 0.00001:
                theta[i][j] = 0
                continue
            if ((el <= 67.5) and (el >= 22.5)) or ((el <= -112.5) and (el >= -168)):  # main diag
                theta[i][j] = 192
            if ((el <= -67.5) and (el >= -112.5)) or ((el <= 112.5) and (el >= 67.5)):  # oy
                theta[i][j] = 64
            if ((el <= 22.5) and (el >= -22.5)) or ((el <= -168) or (el >= 168)):  # ox
                theta[i][j] = 128
            if ((el <= -22.5) and (el >= -67.5)) or ((el <= 168) and (el >= 112.5)):  # another diag
                theta[i][j] = 255

    return theta


@jit(nopython=True)
def nonmax_suppresion(im1_mag, theta):
    """
    Returns image non maxed
    """
    mask = np.ones(theta.shape)
    for i in range(1, theta.shape[0] - 1):
        for j in range(1, theta.shape[1] - 1):
            if im1_mag[i][j] == 0:
                continue
            el = theta[i][j]
            if el == 64:
                if (im1_mag[i][j - 1] > im1_mag[i][j]) or (im1_mag[i][j + 1] > im1_mag[i][j]):
                    mask[i][j] = 0
            if el == 128:
                if (im1_mag[i - 1][j] > im1_mag[i][j]) or (im1_mag[i + 1][j] > im1_mag[i][j]):
                    mask[i][j] = 0
            if el == 192:
                if (im1_mag[i + 1][j + 1] > im1_mag[i][j]) or (im1_mag[i - 1][j - 1] > im1_mag[i][j]):
                    mask[i][j] = 0
            if el == 255:
                if (im1_mag[i + 1][j - 1] > im1_mag[i][j]) or (im1_mag[i - 1][j + 1] > im1_mag[i][j]):
                    mask[i][j] = 0

    print(np.sum(mask), mask.shape[0] * mask.shape[1])
    im1_mag_masked = np.multiply(im1_mag, mask)
    return im1_mag_masked


# TODO: think about norming image? does it work at all?
def norm_image(image, norm=255.0) -> np.array:
    """

    max = norm, min = 0;
    I_new = (I - min(I)) / (max(I) - min(I)) * (max - min) + min =
    = (I - min(I)) / (max(I) - min(I)) * max
    """
    image = image.astype('float')
    return (image - image.min()) / (image.max() - image.min()) * norm


def solve_quadratic_equation(a, b, c):
    """
    Solve quadratic equation
    a * x^2  + b * x + c = 0
    Solves for discr more than zero
    a, b, c -- numpy arrays
    """
    d = b ** 2 - 4 * a * c
    #     print(d)
    eps = 0.000000001
    if np.all(d >= 0):
        s_d = np.sqrt(d)
        x_1 = (-b + s_d) / (2 * a + eps)
        x_2 = (-b - s_d) / (2 * a + eps)
        return x_1, x_2
    else:
        # print(d, a, b, c)
        print(np.sum(d >= 0))
        print(np.sum(d < 0))
        print('Discriminant is less then 0.')


@jit(nopython=True)
def follow_strong_edge(i, j, strong, edge_pict, weak=None, padding=5):
    edge_pict[i][j] = 255
    found = 0
    for t in range(-1, 2):
        for k in range(-1, 2):
            if (t != 0) or (k != 0):
                if (i + t >= padding) and (i + t < strong.shape[0] - padding) and (j + k >= padding) and (
                        j + k < strong.shape[1] - padding):
                    if (strong[i + t][j + k] == 255) and (edge_pict[i + t][j + k] == 0):
                        found += 1
                        follow_strong_edge(i + t, j + k, strong, edge_pict, weak)

    follow_weak_edge(i, j, weak, edge_pict, strong)


@jit(nopython=True)
def follow_weak_edge(i, j, weak, edge_pict, strong, padding=5):
    found = 0
    for t in range(-1, 2):
        for k in range(-1, 2):
            if (t != 0) or (k != 0):
                if (i + t >= padding) and (i + t < strong.shape[0] - padding) and (j + k >= padding) and (
                        j + k < strong.shape[1] - padding):
                    if (weak[i + t][j + k] != 0) and (
                            edge_pict[i + t][j + k] == 0):  # and (strong[i + t][j + k] == 0):
                        #                         print(edge_pict[i + t][j + k], strong[i + t][j + k], i + t, j + k)
                        #                         assert edge_pict[i + t][j + k] == strong[i + t][j + k]
                        found += 1
                        edge_pict[i + t][j + k] = 255
                        follow_weak_edge(i + t, j + k, weak, edge_pict, strong)


@jit(nopython=True)
def follow_edges(strong, weak, padding):
    sh = strong.shape
    edge_pict = np.zeros(shape=sh)
    for i in range(padding, sh[0] - padding):
        for j in range(padding, sh[1] - padding):
            if strong[i][j] != 0:
                # if (140 < i < 200) and (290 < j < 310):
                # print("Found something")
                follow_strong_edge(i, j, strong, edge_pict, weak)

    return edge_pict


# TODO: think about 1 or 255 is max
def my_canny(image,
             sigma_sm: float = 2.0,
             sigma_grad: float = 2.5,
             tau1: float = 0.5,
             tau2: float = 10.0,
             theta=None,
             m_percent=0.7):
    """
    can compute the standard Canny gradient and any supplemented with theta and mag.

    :param image: im_mag or just image
    :param sigma_sm: sigma for smoothing
    :param sigma_grad: sigma for gradient
    :param tau1: first bound
    :param tau2: second bound
    :param theta: direction of gradient
    :return: cannied image
    """
    if theta is None:
        im1 = rgb2gray(image).astype('float')

        im1_smoothed = im1

        g_y = sep_convolve(im1_smoothed, sigma=sigma_grad, order_y=1, order_x=0)
        g_x = sep_convolve(im1_smoothed, sigma=sigma_grad, order_y=0, order_x=1)

        im1_mag = np.sqrt(g_x ** 2 + g_y ** 2)
        print(g_x.min(), g_x.max())
        print(g_y.min(), g_y.max())
        print(g_x.shape)

        im1_mag = norm_image(im1_mag, norm=255.0)

        theta = np.arctan2(g_y, g_x)
        print(theta.min(), theta.max())

        # TODO: add argument
        if False:
            plt.title('im1_mag')
            plt.imshow(im1_mag, cmap='gray')
            plt.show()
            plt.title('theta')
            plt.imshow(theta, cmap='hsv')
            plt.show()

        print('gradient: ', im1_mag.shape)
        print('theta: ', theta.shape)
    else:
        im1_mag = image  # if theta is not None, then image is mag_image

    if theta.max() < np.pi + 1:  # if theta is already normed -- skip
        theta *= 180 / np.pi

    theta = discretize_theta(im1_mag, theta)
    im1_mag_masked = nonmax_suppresion(im1_mag, theta)
    edge_weak = np.count_nonzero(im1_mag_masked)
    values, counts = np.unique(im1_mag_masked, return_counts=True)

    edge = values[1]
    print('begin iterating from: ', edge)
    k = 0
    while np.sum(counts[values > edge]) > edge_weak * m_percent:
        edge += 0.1
        k += 1

    print('number of iterations: ', k)
    print('found second edge: ', edge)

    im = im1_mag_masked
    im_t1 = im.copy()  # weak
    im_t2 = im.copy()
    im_t2[im < edge] = 0  # strong
    im_t2[im >= edge] = 255

    weak = im_t1
    strong = im_t2
    padding = 3
    edge_pict = follow_edges(strong, weak, padding)

    return edge_pict


def sep_convolve(image, sigma=20.0, order_y=1, order_x=1):
    """
    Order is right. In reality order_x is order_y is right.
    """
    # print(image.shape)
    im1 = gaussian_filter1d(image, sigma, axis=0, order=order_y)
    # im1 = norm_image(im1, norm=1.0)
    im2 = gaussian_filter1d(im1, sigma, axis=1, order=order_x)
    # print(im2.shape)
    return im2


def create_gauss_2(sigma):
    sigma = float(sigma)
    n = round(3 * sigma)
    print(sigma, n)

    if n == 0:
        n = 1

    sh = 2 * n + 1
    y, x = np.mgrid[-sh // 2:sh // 2 + 1, -sh // 2:sh // 2 + 1]
    y = -y.astype('float')
    x = x.astype('float')
    g_v = 1.0 / (2 * np.pi * sigma * sigma) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    G_x = -x * g_v / (sigma ** 2)
    G_y = -y * g_v / (sigma ** 2)
    G_xx = (x * x / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
    G_yy = (y * y / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
    G_xy = x * y * g_v / (sigma ** 4)
    return G_x, G_y, G_xx, G_yy, G_xy


def create_steer_filter_1(image, sigma, make_otsu_thresh=True):
    """
    make otsu thresholding.
    """
    # image = img_as_float(image)
    # image = image.astype('float')

    alp_22 = - np.sqrt(3 / 4 / np.pi) * sigma
    alp_20 = - alp_22 / 3

    g_x, g_y, g_xx, g_yy, g_xy = create_gauss_2(sigma)

    im_xx = convolve2d(image, g_xx, mode='same')
    im_yy = convolve2d(image, g_yy, mode='same')
    im_xy = convolve2d(image, g_xy, mode='same')

    # im_xx = sep_convolve(image, sigma=sigma, order_y=0, order_x=2)
    # im_yy = sep_convolve(image, sigma=sigma, order_y=2, order_x=0)
    # im_xy = sep_convolve(image, sigma=sigma, order_y=1, order_x=1)

    print(im_xx.min(), im_xx.max())
    print(im_yy.min(), im_yy.max())
    print(im_xy.min(), im_xy.max())
    # #
    # print('shape g_xx, g_xy: ', im_xx.shape, im_xy.shape)
    # im_xx = norm_image(im_xx, norm=1.0)
    # im_xy = norm_image(im_xy, norm=1.0)
    # im_yy = norm_image(im_yy, norm=1.0)
    print(im_xx.min(), im_xx.max())

    q_1 = alp_20 * im_xx + alp_22 * im_yy
    q_2 = 2 * (alp_20 - alp_22) * im_xy
    q_3 = alp_20 * im_yy + alp_22 * im_xx

    print('qs: ', np.mean(q_1), np.mean(q_2), np.mean(q_3))

    # TODO: find more stable way
    # TODO: or solve another equation
    sols = solve_quadratic_equation(a=q_2, b=(2 * q_1 - 2 * q_3), c=-q_2)

    whole_sol = []
    for sol in sols:
        arc_sol = np.arctan(sol)
        whole_sol.append(arc_sol)
        # whole_sol.append(arc_sol + np.pi)
        # whole_sol.append((arc_sol + np.pi) % (2 * np.pi) - np.pi)
        # whole_sol.append((np.pi + arc_sol) * (arc_sol < 0) + (arc_sol - np.pi) * (arc_sol > 0))

    whole_thetas = np.array(whole_sol)
    # (whole_thetas - 2 * np.pi) * (whole_thetas > np.pi)
    print('whole_thetas: ', whole_thetas.shape)

    whole_image = (np.cos(whole_thetas) ** 2) * q_1 + np.cos(whole_thetas) * \
                  np.sin(whole_thetas) * q_2 + (np.sin(whole_thetas) ** 2) * q_3

    ridge_image = whole_image.max(axis=0)

    print('max, min of ridge image: ', ridge_image.max(), ridge_image.min())

    ridge_image = norm_image(ridge_image)

    print('max, min of ridge image: ', ridge_image.max(), ridge_image.min())

    # return ridge_image, []

    max_args = whole_image.argmax(axis=0)
    sh = max_args.shape
    x, y = np.indices(dimensions=(sh[0], sh[1]))
    max_thetas = whole_thetas[max_args, x, y]
    #
    # max_args = whole_image.argmax(axis=0)
    # max_thetas = np.zeros(shape=max_args.shape)
    # sh = max_args.shape
    # for i in range(sh[0]):
    #     for j in range(sh[1]):
    #         max_thetas[i, j] = whole_thetas[max_args[i, j], i, j]

    im = ridge_image.copy()

    print(np.count_nonzero(im))
    print(im.max(), im.min())

    if make_otsu_thresh:
        thresh = filters.threshold_otsu(im)
        print(thresh)
        im[im < thresh] = 0
        print(np.count_nonzero(im))

    return im, max_thetas


# def get_pairs():
#
# image = get_picture_s('./../ImageProc1/img/old_lena.bmp', to_gray=True)[0]
#
# # image, theta = create_steer_filter_1(t[i], sigma=2.0)
# ridge_image = my_canny(image=image)
#
# fig = plt.figure(figsize=(25, 25))
# fig.add_subplot(1, 2, 1)
# plt.imshow(ridge_image, cmap='gray')
# plt.title('ridge_image')
# fig.add_subplot(1, 2, 2)
# plt.imshow(image, cmap='gray')
# plt.title('original image')
# fig.savefig(f'./image_lena.png', bbox_inches='tight')

class TiffVideo:
    period_fire_frames = 5
    path_to_results = Path('./results')
    path_to_results.mkdir(exist_ok=True)

    def __init__(self, params):
        self.params = params
        self.path_to_results = self.path_to_results / params['file_path'].stem
        print(self.path_to_results)
        self.path_to_results.mkdir(exist_ok=True)

        shutil.copy(params['file_path'], self.path_to_results / f'orig{params["file_path"].suffix}')

        if params['to_load']:
            self._load_data(params['file_path'])
        else:
            self._process_file(params['file_path'])
        self._check_data()
        self._extract_filaments_from_frames(params['processing_frame'])

    def _check_data(self):
        # TODO: check that all size are equal!
        pass

    def _extract_filaments_from_frames(self, processing_type):
        # self.results is binarized images.
        self.video = Video(self.results, self.fire_frames, processing_type)
        with open(self.path_to_results / 'save.pkl', 'wb') as f:
            pkl.dump(self.video, f)

    def _process_file(self, file_path):
        from fire import process_pict_for_fire, get_clusters_from_image

        self.result_file = self._get_info_from_file_path(file_path)
        self.t = tifffile.imread(str(file_path))  # or tiff?

        # self.t = self.t[10:15]

        self.fire_frames = []
        num_of_fire_frames = self.t.shape[0] // self.period_fire_frames
        for i in range(num_of_fire_frames):
            # if i == 5:
            #     print(i)
            pict = self.t[i * self.period_fire_frames:(i + 1) * self.period_fire_frames, :, :]
            processed_pict = process_pict_for_fire(pict)
            # processed_pict = processed_pict[300:500, 120:350]
            clusters = get_clusters_from_image(processed_pict)
            self.fire_frames.append(clusters)

        print(f'number of photos: {self.t.shape}')

        self.results = []
        for i in list(range(self.t.shape[0])):
            image, theta = create_steer_filter_1(self.t[i], sigma=self.params['sigma'])
            ridge_image = my_canny(image=image, theta=theta, m_percent=self.params['m_percent'])

            ridge_image[ridge_image > 0] = 1
            ridge_image = skeletonize(ridge_image).astype('int')
            ridge_image[ridge_image > 0] = 255

            lighted_image = np.zeros(shape=[ridge_image.shape[0], ridge_image.shape[1], 3])
            lighted_image[:, :, 0] = ridge_image
            lighted_image[:, :, 1] = self.t[i]
            lighted_image[:, :, 2] = self.t[i]

            # lighted_image = lighted_image[350:460, 0:300, :]

            self.results.append(lighted_image)

        images = np.stack(self.results, axis=0)
        print(f'number of photos after feature finding: {images.shape[0]}')
        #  Doesn't work!!!
        tifffile.imsave(str(self.result_file), data=images.astype('uint8'))

    def _load_data(self, file_path):
        self.result_file = self._get_info_from_file_path(file_path)
        self.t = tifffile.imread(str(file_path))
        self.results = tifffile.imread(str(self.result_file))

    @staticmethod
    def _get_info_from_file_path(file_path):
        file_name = file_path.name
        folder_name = file_path.stem
        directory = Path.cwd() / 'results' / folder_name
        directory.mkdir(parents=True, exist_ok=True)  # exist_ok?, iterate?
        result_file = directory / file_name
        return result_file

    def save_pairs_to_pdf_file(self, mode='1'):
        """

        :param mode: can be 1 or 2
        :return:
        """
        sh = self.results[0].shape[0]  # assume that all shapes are equal!!!
        sh_big = self.t[0].shape[0]
        pad = (sh_big - sh) // 2
        if mode != '1':
            with open('./save.pkl', 'rb') as f:
                vid = pkl.load(f)
            img = np.zeros(shape=[len(vid.frames), vid.width, vid.height, 3])
            for frame_num, frame in enumerate(vid.frames):
                for _, filament in enumerate(frame.filaments):
                    img[frame_num][filament.xs, filament.ys, 1:3] = 255
                    img[frame_num][filament.center_x, filament.center_y, 0] = 255
            # tifffile.imsave(str(self.result_file), data=images.astype('uint8'))

        for ind, lighted_image in enumerate(self.results):
            if mode == '1':
                ridge_image = lighted_image[:, :, 0]
                ridge_image_temp = np.zeros_like(self.t[0])
                ridge_image_temp[pad:sh + pad, pad:sh + pad] = ridge_image
                ridge_image_temp[ridge_image_temp > 0] = 255
            else:
                ridge_image_temp = img[ind][:, :, 1]
                ridge_image = img[ind][:, :, 1]

            w, h = figaspect(ridge_image_temp.shape[0] / (ridge_image_temp.shape[1] * 2))
            fig = plt.figure(num=ind, figsize=(w, h))
            fig.set_dpi(400)
            fig.add_subplot(1, 2, 1)
            plt.imshow(ridge_image, cmap='gray')
            plt.title('ridge_image')
            fig.add_subplot(1, 2, 2)
            plt.imshow(self.t[ind], cmap='gray')
            plt.title('original image')
            plt.tight_layout()

        # TODO: add option to add several pdfs. parse all pdfs in directory, choose max number, add 1, use this one
        pdf_path = '.'.join(str(self.result_file).split('.')[:-1]) + '.pdf'
        pdf = PdfPages(pdf_path)
        for fig_num in plt.get_fignums():
            pdf.savefig(fig_num)
        pdf.close()
        plt.close('all')


def create_argparse():
    parser = argparse.ArgumentParser(description='Process tiff file with filaments for future analysis')
    parser.add_argument('file_path', type=str, help="path to tiff file for analysis")
    parser.add_argument('--sigma', type=float, default=2.0, help='sigma in gaussian')
    parser.add_argument('--m_percent', type=float, default=0.8, help='canny parameter for hysteresis')
    parser.add_argument('--processing_frame', type=str, default='simple', help='does nothing for noe')
    parser.add_argument('--to_load', type=bool, default=False, help='does nothing for now')
    return parser


# TODO: add config! or argparse
# TODO: use only argparse withput dict params
def main(args):
    params = {
        'sigma': args.sigma,
        'to_load': args.to_load,
        'file_path': Path(args.file_path),
        'm_percent': args.m_percent,
        'processing_frame': args.processing_frame
    }

    print(params)

    tf = TiffVideo(params)
    # tf.save_pairs_to_pdf_file(mode='2')


if __name__ == '__main__':
    parser = create_argparse()
    pargs = parser.parse_args()

    main(pargs)
