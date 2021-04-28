import numpy as np
from utils import *
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile

T_in=7


 # print(x_data_padded.shape) (26, 100, 115, 3)
 # print(y_data.shape) (20, 400, 460, 3)
 #####################################################################
def GetVideo(VideoNumber,quality):
    if quality:
        path= 'F:\\Test_HighQuality'
    else:
        path='F:\\Test_LowQuality'

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

#######################################################################
def LoadDataSet(VideoNumber):

    X_dataset=GetVideo(VideoNumber, False)
    Y_dataset=GetVideo(VideoNumber, True)
    y_true = []
    for i in range(len(Y_dataset)):
        YtrueList = []
        for j in range(7):
            YtrueList.append(Y_dataset[i][j][np.newaxis, np.newaxis, :, :, :])  # print(yy[1].shape) (1, 1, 400, 460, 3)
        y_true.append(YtrueList)
    y_true = np.asarray(y_true)
    Y_dataset = y_true
    return X_dataset,Y_dataset
#########################################
x_data, y_data = LoadDataSet(0)
print(x_data.shape)
with tf.Graph().as_default():
    output_graph_def = tf.compat.v1.GraphDef()
    output_graph_path = './model/My_Duf_2.pb'

    with gfile.FastGFile(output_graph_path,"rb") as f:
        output_graph_def.ParseFromString(f.read())
        # fix nodes
        for node in output_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]
        _ = tf.import_graph_def(output_graph_def,name="")

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        input = sess.graph.get_tensor_by_name("L_in:0")
        output = sess.graph.get_tensor_by_name("out_H:0")
        is_train=sess.graph.get_tensor_by_name('is_train:0')

        total_loss=0
        for j in range(x_data.shape[0]):
            in_L = x_data[j]  # select T_in frames
            in_L = in_L[np.newaxis, :, :, :, :]
            y_out = sess.run(output, feed_dict={input: in_L, is_train: False})
            Image.fromarray(np.around(y_out[0, 0] * 255).astype(np.uint8)).save('./result_test/{:05}.png'.format(j))

            cost = Huber(y_true=y_data[j], y_pred=y_out, delta=0.01)
            loss = sess.run(cost)
            total_loss = total_loss+loss
            print('this single test cost: {:.7f}'.format(loss))

        avg_test_loss=total_loss/x_data.shape[0]
        print("avg test cost: {:.7f}".format(avg_test_loss))
