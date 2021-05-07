import os
from Imports import *
from Imports import *
from utils import LoadDataSet


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Ahmed1")
# model=[]
stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
# Size of input temporal radius
T_in = 7
# Upscaling factor
R = 2

model = keras.models.load_model('D:\semestre9\gp\GraduationProject\SuperResolutoin\My Model9')
path = 'New folder'
print("Ahmed2")

for video in range(10):
    print("Ahmed3")
    x_train_All, y_train_All = LoadDataSet(video)
    for every7 in range(len(x_train_All)):
        x_train = x_train_All[every7]
        x_train = x_train[np.newaxis, :, :, :, :]
        print(x_train.shape, "x_train.shape")
        image = model.predict(np.array(x_train))
        # print("image[0, 0].shape", image[0, 0].shape)
        # x = tf.squeeze(image)  # b, h, w, R*R
        Image.fromarray(np.around(image[0, 0]*255).astype(np.uint8)).save('./result_test/{:05}.png'.format(every7))

