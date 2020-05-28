from fil_processing import find_filaments, Filament
import numpy as np
from skimage.feature import canny
from matplotlib import pyplot as plt


def draw_first_filament():
    img = np.zeros(shape=[20, 20])
    img[10, 10] = 255
    img[11, 11] = 255
    img[12, 12] = 255
    img[13, 13] = 255
    return img


def draw_mult_filaments():
    img = np.zeros(shape=[20, 20])
    img[10, 10] = 255
    img[11, 11] = 255
    img[12, 12] = 255
    img[13, 13] = 255

    img[5, 5] = 255
    img[5, 6] = 255
    img[5, 7] = 255
    img[5, 8] = 255
    return img


def draw_second_filament():
    img = np.zeros(shape=[20, 20])
    img[10, 10] = 255
    img[11, 11] = 255
    img[12, 12] = 255
    img[13, 13] = 255
    img[9, 9] = 255
    img[8, 8] = 255
    img[7, 7] = 255
    img[6, 6] = 255
    return img


def test_1():
    fil1 = draw_first_filament()
    print(find_filaments(fil1))

    fil2 = draw_mult_filaments()
    print(find_filaments(fil2))

    fil3 = draw_second_filament()
    print(find_filaments(fil3))


def test_2():
    coords = np.array([[10, 10], [11, 11], [12, 12], [13, 13]])
    fil1 = Filament(coords)
    print(fil1.number_of_tips)
    coords = np.array([[11, 11], [12, 12], [13, 13], [13, 14], [13, 15]])
    fil2 = Filament(coords)
    print(fil2.number_of_tips)
    coords = np.array([[11, 11], [12, 12], [13, 13], [13, 14], [13, 15]])
    fil3 = Filament(coords[::-1])
    print(fil3.number_of_tips)
    print(Filament.fils_distance_fast(fil1, fil2))
    print(Filament.overlap_score_fast(fil1, fil2))
    print(Filament.fils_distance_fast(fil2, fil3))
    print(Filament.overlap_score_fast(fil2, fil3))

def test_3():
    ideal_edge = np.ones(shape=[20, 20])
    xs = np.arange(10, 20)
    ys = np.arange(10, 20)
    values = np.meshgrid(xs, ys)
    ideal_edge[values] = 0
    plt.imshow(ideal_edge)
    plt.show()
    print(ideal_edge)
    res = canny(ideal_edge, sigma=0.00001)
    print(canny(ideal_edge, sigma=0.00001))
    plt.figure(dpi=500)
    plt.xticks(np.arange(20))
    plt.yticks(np.arange(20))
    plt.imshow(res)
    plt.imsave('./1.png', res)
    plt.show()


def main():
    d = {
        "a": 1,
        "b": 2,
        "c": 3
    }

    for k, v in d.items():
        print(k, v)
        if k == 'b':
            d[k] = 20

    print(d)

    a = [*[1, 2, 3], *[4, 5, 6]]
    print(a)

    print(len([el for el in []]))


if __name__ == "__main__":
    main()