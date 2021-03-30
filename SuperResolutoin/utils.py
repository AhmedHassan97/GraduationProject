from Imports import *


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


# Gaussian kernel for downsampling
def gkern(kernlen=13, nsig=1.6):
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def getMotionMatrix(frame1, frame2):
    frame1_Gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_Gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(frame1_Gray, frame2_Gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    sqFlow1 = np.square(flow[..., 0])
    sqFlow2 = np.square(flow[..., 1])
    sumFlow = sqFlow1 + sqFlow2
    mag_matrix = np.sqrt(sumFlow)
    return mag_matrix


def get144OfMostMotion(mag_matrix):
    maxAverage = 0
    i_pixel = 0
    j_pixel = 0
    for k in range(mag_matrix.shape[0]):
        for j in range(mag_matrix.shape[1]):
            if k + 144 <= mag_matrix.shape[0] and j + 144 <= mag_matrix.shape[1]:
                avg = np.mean(mag_matrix[k:k + 144, j:j + 144])
                if avg > maxAverage:
                    maxAverage = avg
                    i_pixel = k
                    j_pixel = j

    return i_pixel, j_pixel
