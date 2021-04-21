from Imports import *

HighQualityPath = 'F:\\Dataset_HighQuality'
LowQualityPath= 'F:\\Dataset_LowQuality'
def convolve2D(image, kernel, padding=6, strides=1):
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


def LoadImage(path, color_mode='RGB', channel_mean=None, modcrop=[0, 0, 0, 0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.

    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    ## Load image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:, :, 0:1]

    ## To 0-1
    x *= 1.0 / 255.0

    if channel_mean:
        x[:, :, 0] -= channel_mean[0]
        x[:, :, 1] -= channel_mean[1]
        x[:, :, 2] -= channel_mean[2]

    if modcrop[0] * modcrop[1] * modcrop[2] * modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]

    return x


def get_x():
    T_in = 7
    path = 'F:\\Dataset_LowQuality'
    dir_frames = glob.glob(path + "*.jpg")
    dir_frames.sort()
    frames = []
    for f in dir_frames:
        frames.append(LoadImage(f))
    frames = np.asarray(frames)

    return frames


def get_y():
    path = 'F:\\Dataset_HighQuality'
    dir_frames = glob.glob(path + "*.jpg")
    dir_frames.sort()
    frames = []
    for f in dir_frames:
        frames.append(LoadImage(f))
    frames = np.asarray(frames)

    return frames


def Conv3D(input, kernel_shape, strides, padding, name='Conv3d', W_initializer=tf.compat.v1.initializers.he_uniform(),
           bias=True):
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable("W", kernel_shape, initializer=W_initializer)
        if bias is True:
            b = tf.compat.v1.get_variable("b", (kernel_shape[-1]), initializer=tf.constant_initializer(value=0.0))
        else:
            b = 0

    return tf.nn.conv3d(input, W, strides, padding) + b


def Huber(y_true, y_pred, delta, axis=None):
    abs_error = tf.abs(y_pred - y_true)
    quadratic = tf.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses, axis=axis)


def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):
    '''
    https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
    '''
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops

    axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]

    with tf.compat.v1.variable_scope(name):
        beta = tf.compat.v1.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
        gamma = tf.compat.v1.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
        moving_mean = tf.compat.v1.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0),
                                                trainable=False)
        moving_variance = tf.compat.v1.get_variable('moving_variance', fdim,
                                                    initializer=tf.constant_initializer(value=0.0),
                                                    trainable=False)

        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(input, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay,
                                                                           zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input, mean, variance, beta, gamma,
                                     1e-3)  # , tf.stack([mean[0], variance[0], beta[0], gamma[0]])


def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], ds_x[4]])

    y = tf.compat.v1.depth_to_space(x, block_size)

    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x


def DynFilter3D(x, F, filter_size):
    '''
    3D Dynamic filtering
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    '''
    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)),
                                       (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    filter_localexpand = tf.Variable(filter_localexpand_np, trainable=False, dtype='float32', name='filter_localexpand')
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    x_localexpand = tf.nn.conv2d(x, filter_localexpand, [1, 1, 1, 1], 'SAME')  # b, h, w, 1*5*5
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, h, w, 1, 1*5*5
    x = tf.matmul(x_localexpand, F)  # b, h, w, 1, R*R
    x = tf.squeeze(x, axis=3)  # b, h, w, R*R

    return x


def freeze_graph(check_point_folder, model_folder, pb_name):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(check_point_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    output_graph = model_folder + '/' + pb_name
    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    output_node_names = "out_H"
    list_str = []

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.compat.v1.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # fix batch norm nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.compat.v1.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

def GetVideo(VideoNumber,quality):
    if quality:
        path= HighQualityPath
    else:
        path=LowQualityPath

    files = os.listdir(path)
    files.sort()

    Video=os.listdir(path + '\\' + files[VideoNumber])
    print(Video)
    VideoList=[]
    for i in range(len(Video)):
        imagesList = []
        listOfSeven = os.listdir(path + '\\' + files[VideoNumber] + '\\' + Video[i])
        for j in range(7):
            image=cv2.imread(path + '\\' + files[VideoNumber] + '\\' + Video[i]+'\\' + listOfSeven[j])
            imagesList.append(image)
        VideoList.append(imagesList)

    # print(np.asarray(VideoList)[0][3])
    return np.asarray(VideoList)

