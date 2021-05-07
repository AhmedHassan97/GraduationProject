from Imports import *


def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], ds_x[4]])

    y = tf.compat.v1.depth_to_space(x, block_size)

    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x


stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
# Size of input temporal radius
T_in = 7
# Upscaling factor
R = 4


def conv2d(x, f):
    x = tf.nn.conv2d(x, f, [1, 1, 1, 1], 'SAME')  # b, h, w, 1*5*5
    x = tf.expand_dims(x, axis=3)  # b, h, w, 1, 1*5*5
    return x


def dynFilterBlock(x, Fx, c):
    print(1)
    print(x.shape)
    print(Fx.shape)

    filter_size = [1, 5, 5]

    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)),
                                       (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    x = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 3, 1]))(x)

    x_localexpand = Lambda(conv2d, arguments={"f": filter_localexpand_np})(x)

    # x_local_expand = Conv2D(filter_localexpand_np[-2],filter_localexpand_np[:-2], padding='same',activation=None)(x)
    # print(x_local_expand.shape)
    # ShapeOfFilter=filter_localexpand_np
    # x_local_expand = Conv2D(ShapeOfFilter[-2],ShapeOfFilter[:-2], padding='same',activation=None)
    # c2=x_local_expand(x)
    # x_local_expand.set_weights(filter_localexpand_np)
    # x_local_expand.trainable=False

    x = tf.matmul(x_localexpand, Fx)  # b, h, w, 1, R*R
    print("X Shape:", x.shape)
    x = tf.squeeze(x, axis=3)  # b, h, w, R*R

    return x


def bn_block(input_, F, G, s, clip=None):
    bn1 = BatchNormalization()(input_)
    bn1 = tf.keras.layers.ReLU()(bn1)
    conv1 = Conv3D(F, (1, 1, 1))(bn1)

    bn2 = BatchNormalization()(conv1)
    bn2 = tf.keras.layers.ReLU()(bn2)
    conv2_input = Lambda(lambda x: tf.pad(x, s))(bn2)
    conv2 = Conv3D(G, (3, 3, 3))(conv2_input)
    concat = None
    if clip is None:
        concat = keras.layers.Concatenate(axis=4)([input_, conv2])
    else:
        concat = keras.layers.Concatenate(axis=4)([input_[:, clip[0]:clip[1]], conv2])
    return concat


def reshapeWrapper(x):
    shape = tf.shape(x)
    return tf.reshape(x, [shape[0], shape[1], shape[2], shape[3], 25, 16])


def create_model():
    # input_shape = ((7, 36, 36, 3))
    input_ = Input(shape=(7, None, None, 3))
    padInput = Lambda(lambda x: tf.pad(x, sp))(input_)
    initial_conv = Conv3D(64, (1, 3, 3))(padInput)
    conv_blk = initial_conv
    F = 64
    G = 32
    # First section
    for _ in range(3):
        conv_blk = bn_block(conv_blk, F, G, stp)
        F += G

    # Second section
    for _ in range(3):
        conv_blk = bn_block(conv_blk, F, G, sp, clip=[1, -1])

        F += G
    # Shared section
    conv_blk = BatchNormalization()(conv_blk)
    conv_blk = tf.keras.layers.ReLU()(conv_blk)
    #   conv_blk = ZeroPadding3D()
    conv_blk = Lambda(lambda x: tf.pad(x, sp))(conv_blk)

    conv_blk = Conv3D(256, (1, 3, 3), activation='relu')(conv_blk)

    # R
    r = Conv3D(256, (1, 1, 1), activation='relu')(conv_blk)
    r = Conv3D(16 * 3, (1, 1, 1))(r)
    # F
    f = Conv3D(512, (1, 1, 1), activation='relu')(conv_blk)
    f = Conv3D(5 * 5 * 16, (1, 1, 1))(f)

    f = keras.layers.Lambda(reshapeWrapper)(f)

    f = Softmax(axis=4)(f)

    Fx = f
    Rx = r
    x = input_
    x_c = []
    # c refers to channels

    for c in range(3):
        #  x = AnchorsLayer(dynFilterBlock(x,Fx,c), name="x")(x)
        print(Fx.shape, "Fx.shape")
        # t = dynFilterBlock(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], c)
        t = dynFilterBlock(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], c)

        t = tf.compat.v1.depth_to_space(t, R)  # [B,H*R,W*R,1]
        # x=Lambda(dynFilterBlock, arguments={'c': c, 'Fx':Fx})(x)
        x_c += [t]
    print(x_c)
    x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3] Tensor("concat_9:0", shape=(?, ?, ?, 3), dtype=float32)
    x = tf.expand_dims(x, axis=1)  # Tensor("ExpandDims_3:0", shape=(?, 1, ?, ?, 3), dtype=float32)
    Rx = depth_to_space_3D(Rx, R)  # [B,1,H*R,W*R,3] Tensor("Reshape_6:0", shape=(?, ?, ?, ?, ?), dtype=float32)
    x += Rx  # Tensor("add_18:0", shape=(?, ?, ?, ?, 3), dtype=float32)
    out_H = tf.clip_by_value(x, 0, 1, name='out_H')
    print(out_H.shape, "out_H.shape")

    return keras.Model(inputs=[input_], outputs=[out_H])


HighQualityPath = r'C:\Users\Ahmed M. Hassan\OneDrive\Desktop\Dataset_HighQuality'
LowQualityPath = r'C:\Users\Ahmed M. Hassan\OneDrive\Desktop\Dataset_LowQuality'


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


def GetVideo(VideoNumber, quality):
    if quality:
        path = HighQualityPath
    else:
        path = LowQualityPath

    files = os.listdir(path)
    files.sort()

    Video = os.listdir(path + '\\' + files[VideoNumber])
    VideoList = []
    for i in range(len(Video)):
        imagesList = []
        listOfSeven = os.listdir(path + '\\' + files[VideoNumber] + '\\' + Video[i])
        if len(listOfSeven) == 7:
            for j in range(len(listOfSeven)):
                image = LoadImage(path + '\\' + files[VideoNumber] + '\\' + Video[i] + '\\' + listOfSeven[j])
                imagesList.append(image)
            VideoList.append(imagesList)
    # print(np.asarray(VideoList)[0][3])
    return np.asarray(VideoList)


def LoadDataSet(VideoNumber):
    X_dataset = GetVideo(VideoNumber, False)
    Y_dataset = GetVideo(VideoNumber, True)
    y_true = []
    for i in range(len(Y_dataset)):
        YtrueList = []
        for j in range(7):
            y_image = Y_dataset[i][j]
            YtrueList.append(Y_dataset[i][j][np.newaxis, np.newaxis, :, :, :])  # print(yy[1].shape) (1, 1, 400, 460, 3)
        y_true.append(YtrueList)

    y_true = np.asarray(y_true)
    Y_dataset = y_true
    return X_dataset, Y_dataset
