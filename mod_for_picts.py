import numpy as np
from skimage.io import imread, imsave
import sys
import scipy.signal


def get_picture_s(in_image1, in_image2=None, to_gray=True):
    im1 = imread(in_image1)
    im1 = im1.astype('float')
    if len(im1.shape) == 2:
        im1 = np.expand_dims(im1, axis=2)

    if in_image2 is not None:
        im2 = imread(in_image2)
        im2 = im2.astype('float')
        if len(im2.shape) == 2:
            im2 = np.expand_dims(im2, axis=2)
        assert im1.shape == im2.shape
    else:
        im2 = None

    if to_gray:
        assert (len(im1.shape) == 3) and (im1.shape[2] == 3)
        r, g, b = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
        im1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
        if im2 is not None:
            assert (len(im2.shape) == 3) and (im2.shape[2] == 3)
            r, g, b = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]
            im2 = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return im1, im2


def even(s, s_new, n):

    for j in range(0, n):
        for i in range(n, s.shape[0] + n):
            s_new[i, s.shape[1] + n + j] = 2 * s_new[i, s.shape[1] + n - 1] - s_new[i, (s.shape[1] + n - 2 - j)]

    for j in range(0, n):
        for i in range(n, s.shape[0] + n):
            s_new[i, n - j - 1] = 2 * s_new[i, n] - s_new[i, (n + j + 1)]

    for i in range(0, n):
        for j in range(0, s.shape[1] + 2 * n):
            s_new[n - i - 1, j] = 2 * s_new[n, j] - s_new[n + 1 + i, j]

    for i in range(0, n):
        for j in range(0, s.shape[1] + 2 * n):
            s_new[s.shape[0] + n + i, j] = 2 * s_new[s.shape[0] + n - 1, j] - s_new[s.shape[0] + n - 2 - i, j]

    return s_new


def odd(s, s_new, n):
    for j in range(0, n):
        for i in range(n, s.shape[0] + n):
            s_new[i, s.shape[1] + n + j] = s_new[i, (s.shape[1] + n - 2 - j)]

    for j in range(0, n):
        for i in range(n, s.shape[0] + n):
            s_new[i, n - j - 1] = s_new[i, (n + j + 1)]

    for i in range(0, n):
        for j in range(0, s.shape[1] + 2 * n):
            s_new[n - i - 1, j] = s_new[n + 1 + i, j]

    for i in range(0, n):
        for j in range(0, s.shape[1] + 2 * n):
            s_new[s.shape[0] + n + i, j] = s_new[s.shape[0] + n - 2 - i, j]

    return s_new


def rep(s, s_new, n):

    s_new[n:-n, 0:n, :] = np.expand_dims(s_new[n:-n, n, :], axis=1)
    s_new[n:-n, s.shape[1] + n:s.shape[1] + 2 * n, :] = np.expand_dims(s_new[n:-n, s.shape[1] + n - 1, :], axis=1)
    s_new[:n, n:-n, :] = np.expand_dims(s_new[n, n:s.shape[1] + n, :], axis=0)
    s_new[s.shape[0] + n:, n:-n, :] = np.expand_dims(s_new[s.shape[0] + n - 1, n:s.shape[1] + n, :], axis=0)
    s_new[0:n, 0:n, :] = np.expand_dims(np.expand_dims(s_new[n, n, :], axis=0), axis=0)
    s_new[-n:, -n:, :] = np.expand_dims(np.expand_dims(s_new[-n - 1, -n - 1, :], axis=0), axis=0)
    s_new[-n:, 0:n, :] = np.expand_dims(np.expand_dims(s_new[-n - 1, n, :], axis=0), axis=0)
    s_new[:n, -n:, :] = np.expand_dims(np.expand_dims(s_new[n, -n - 1, :], axis=0), axis=0)

    return s_new


# def convolve2d(s, filters, make_pos=True):
#     # gets 3d matrix (or maybe two and makes some thing)
#     # matrix is already padded
#     # returns smaller matrix
#     if not isinstance(filters, (tuple, list)):  # it does mean that we got only one filter. need to make tuple of it.
#         filters = tuple([filters])
#     n = filters[0].shape[0]  # later change, assumed filters of the same size
#     if len(s.shape) == 2:  # broadcasting image if it's greyscale
#         s = np.expand_dims(s, axis=2)
#     number_of_channels = s.shape[2]
#     # make list of output matrices which are of the real size of an image in input
#     res = [np.zeros(shape=[s.shape[0] - 2 * (n // 2), s.shape[1] - 2 * (n // 2), s.shape[2]]) for _ in filters]
#     # filter of size N
#     padd = n // 2  # this is how many padding pixels around a picture
#     for k in range(number_of_channels):
#         s_l = s[:, :, k]  # get layer
#         print(s_l.shape)
#         for i in range(padd, s_l.shape[0] - padd):
#             for j in range(padd, s_l.shape[1] - padd):
#                 for ind, f in enumerate(filters):
#                     for t in range(-padd, padd + 1):
#                         for l in range(-padd, padd + 1):
#                             res[ind][i - padd, j - padd, k] += s_l[i + t, j + l] * f[-(t + padd), -(l + padd)]
#
#                     if make_pos and (res[ind][i - padd, j - padd, k] < 0):
#                         res[ind][i - padd, j - padd, k] *= -1
#
#     if res[0].shape[2] == 1:  # downcasting image if it's greyscale
#         for ind, _ in enumerate(res):
#             res[ind] = res[ind].squeeze(2)
#
#     # TODO: it is bug, change later
#     return res


def expand_image(s, n, extr='rep'):
    s = s.astype(dtype='float')
    if len(s.shape) == 2:
        s = np.expand_dims(s, axis=2)

    s_new = np.zeros(shape=[s.shape[0] + 2 * n, s.shape[1] + 2 * n, s.shape[2]])
    s_new[n:s.shape[0] + n, n:s.shape[1] + n] = s
    
    if extr == 'rep':
        s_new = rep(s, s_new, n)
    elif extr == 'even':
        s_new = even(s, s_new, n)
    else:
        print('Using for default odd')
        s_new = odd(s, s_new, n)

    if s_new.shape[2] == 1:
        s_new = s_new.squeeze(2)
    return s_new


def gauss(extr, sigma, s):
    sigma = float(sigma)
    n = round(3 * sigma)
    if n == 0:
        n = 1
    sh = 2 * n + 1
    G = np.zeros(shape=[sh, sh])
    for i in range(sh):
        for j in range(sh):
            x = i - sh // 2  # make 3 sigma
            y = j - sh // 2
            G[i, j] = 1.0 / (2 * np.pi * sigma * sigma) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    G = G / np.sum(G)
    s = s.astype('float')
    s = expand_image(s, n, extr)
    res = scipy.signal.convolve2d(s, G, mode='valid')
    return res


# def create_gauss(sigma):
#     sigma = float(sigma)
#     n = round(3 * sigma)
#     print(sigma, n)
#
#     if n == 0:
#         n = 1
#
#     sh = 2 * n + 1
#     G_x = np.zeros(shape=[sh, sh])
#     G_y = np.zeros(shape=[sh, sh])
#     G_yy = np.zeros(shape=[sh, sh])
#     G_xx = np.zeros(shape=[sh, sh])
#     G_xy = np.zeros(shape=[sh, sh])
#     for i in range(sh):
#         for j in range(sh):
#             y = sh // 2 - i  # make 3 sigma
#             x = j - sh // 2
#     #       TODO: think about sigma and constants!!!
#             g_v = 1.0 / (2 * np.pi * sigma * sigma) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
#             G_x[i, j] = float(-x) * g_v / (sigma ** 2)
#             G_y[i, j] = float(-y) * g_v / (sigma ** 2)
#             G_xx[i, j] = (x * x / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
#             G_yy[i, j] = (y * y / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
#             G_xy[i, j] = x * y * g_v / (sigma ** 4)
#     return G_x, G_y, G_xx, G_yy, G_xy

def create_gauss(sigma):
    sigma = float(sigma)
    n = round(3 * sigma)
    #     print(sigma, n)

    if n == 0:
        n = 1

    sh = 2 * n + 1
    if sh % 2 == 0:
        y, x = np.mgrid[-sh // 2 + 1:sh // 2 + 1, -sh // 2 + 1:sh // 2 + 1] - 1
    else:
        y, x = np.mgrid[-sh // 2 + 1:sh // 2 + 1, -sh // 2 + 1:sh // 2 + 1]

    y = -y.astype('float')
    x = x.astype('float')
    g_v = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
    G_x = -x * g_v / (sigma ** 2)
    G_y = -y * g_v / (sigma ** 2)
    G_xx = (x * x / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
    G_yy = (y * y / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
    G_xy = x * y * g_v / (sigma ** 4)

    return G_x, G_y, G_xx, G_yy, G_xy


    # y, x = np.mgrid[-sh//2:sh//2+1,-sh//2:sh//2+1]
    # y = -y.astype('float')
    # x = x.astype('float')
    # g_v = 1.0 / (2 * np.pi * sigma * sigma) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # G_x = -x * g_v / (sigma ** 2)
    # G_y = -y * g_v / (sigma ** 2)
    # G_xx = (x * x / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
    # G_yy = (y * y / (sigma ** 2) - 1.0) * g_v / (sigma ** 2)
    # G_xy = x * y * g_v / (sigma ** 4)
    # return G_x / np.sum(G_x), G_y / np.sum(G_y), G_xx / np.sum(G_xx), G_yy / np.sum(G_yy), G_xy / np.sum(G_xy)


def gradient(extr, sigma, s):
    sigma = float(sigma)
    n = round(3 * sigma)
    print('sigma: ' + str(sigma), 'n: ' + str(n))
    if n == 0:
        n = 1

    G_x, G_y, _, _, _ = create_gauss(sigma)

    s = s.astype('float')
    s = expand_image(s, n, extr)
    res = [0, 0]
    # res = convolve2d(s, (G_x, G_y), make_pos=False)
    res[0] = scipy.signal.convolve2d(s, G_x, mode='valid')
    res[1] = scipy.signal.convolve2d(s, G_y, mode='valid')
    mag = np.sqrt(res[1] ** 2 + res[0] ** 2)
    # print(mag.min(), mag.max())
    mag = (255.0 / mag.max()) * mag
    # print(mag.min(), mag.max())
    # print(G_x.min(), G_x.max())
    # print(G_y.min(), G_y.max())
    G_x = (255.0 / res[0].max()) * res[0]
    G_y = (255.0 / res[1].max()) * res[1]
    # print(G_x.min(), G_x.max())
    # print(G_y.min(), G_y.max())
    theta = np.arctan2(G_y, G_x)
    # print(theta.shape)
    # print(G_x.shape, G_y.shape)
    return mag, theta, G_x, G_y