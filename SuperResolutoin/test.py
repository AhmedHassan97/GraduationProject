from Imports import *
import tensorflow as tf
#
#
def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)



# W = tf.constant(gkern())
# depthwise_F = tf.tile(W, [1, 1])
#
# print(depthwise_F)
io.imshow(gkern().shape)
io.show()

# a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
# b = tf.constant([2,3], tf.int32)
# print(tf.tile(a, b))
#
# filter_height, filter_width = 13, 13
# pad_height = filter_height - 1
# pad_width = filter_width - 1
#
# # When pad_height (pad_width) is odd, we pad more to bottom (right),
# # following the same convention as conv2d().
# pad_top = pad_height // 2
# pad_bottom = pad_height - pad_top
# pad_left = pad_width // 2
# pad_right = pad_width - pad_left
# pad_array = [[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]]
# path="Frame 001.png"
# img=io.imread(path)
# new=tf.pad(img, pad_array, mode='REFLECT')
# io.imshow(new)
# io.show()