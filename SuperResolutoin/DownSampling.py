from Imports import *


def quarter_res_avg(im):
    original_width = im.shape[1]
    original_height = im.shape[0]
    width = int(original_width / 4)
    height = int(original_height / 4)
    # print(width,height)
    resized_image = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    scale = 4

    for i in range(height):
        for j in range(width):
            temp = np.array([0, 0, 0])
            for x in range(scale):
                for y in range(scale):
                    temp += im[i*scale + x, j*scale + y]
            resized_image[i, j] = temp/(scale*scale)

    return resized_image

