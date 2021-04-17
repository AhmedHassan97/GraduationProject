from Imports import *
from utils import *


tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


# Size of input temporal radius
T_in = 7
# Upscaling factor
R = 4
#get the data set

# X_dataset = get_x()
# Y_dataset = get_y()

X_dataset = GetVideo(0, False)
Y_dataset = GetVideo(0, True)

print(Y_dataset.shape, "Shape of y ")


y_true = []
for i in range(len(Y_dataset)):
    YtrueList = []
    for j in range(7):
        YtrueList.append(Y_dataset[i][j][np.newaxis, np.newaxis, :, :, :])  # print(yy[1].shape) (1, 1, 400, 460, 3)
    y_true.append(YtrueList)
y_true = np.asarray(y_true)
Y_dataset = y_true

H_out_true = tf.compat.v1.placeholder(tf.float32, shape=(1, 1, None, None, 3), name='H_out_true')

is_train = tf.compat.v1.placeholder(tf.bool, shape=[], name='is_train')  # Phase ,scalar

L = tf.compat.v1.placeholder(tf.float32, shape=[None, 7, None, None, None], name='L_in')

stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

x = Conv3D(tf.pad(L, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID', name='conv1')

F = 64
G = 32

for r in range(3):
    t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
    t = tf.nn.relu(t)
    t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

    t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
    t = tf.nn.relu(t)
    t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
               name='Rconv' + str(r + 1) + 'b')

    x = tf.concat([x, t], 4)
    F += G
for r in range(3, 6):
    t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
    t = tf.nn.relu(t)
    t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

    t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
    t = tf.nn.relu(t)
    t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
               name='Rconv' + str(r + 1) + 'b')

    x = tf.concat([x[:, 1:-1], t], 4)
    F += G

# sharen section
x = BatchNorm(x, is_train, name='fbn1')
x = tf.nn.relu(x)
x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')
x = tf.nn.relu(x)

# R
r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
r = tf.nn.relu(r)
r = Conv3D(r, [1, 1, 1, 256, 3 * 16], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

# F
f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')
f = tf.nn.relu(f)
f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * 16], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

ds_f = tf.shape(f)
f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
f = tf.compat.v1.nn.softmax(f, dim=4)

Fx = f
Rx = r

x = L
x_c = []
# c refers to channels
for c in range(3):
    t = DynFilter3D(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
    t = tf.compat.v1.depth_to_space(t, R)  # [B,H*R,W*R,1]
    x_c += [t]

x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3] Tensor("concat_9:0", shape=(?, ?, ?, 3), dtype=float32)
#y_hat

x = tf.expand_dims(x, axis=1)  # Tensor("ExpandDims_3:0", shape=(?, 1, ?, ?, 3), dtype=float32)
Rx = depth_to_space_3D(Rx, R)  # [B,1,H*R,W*R,3] Tensor("Reshape_6:0", shape=(?, ?, ?, ?, ?), dtype=float32)
x += Rx  # Tensor("add_18:0", shape=(?, ?, ?, ?, 3), dtype=float32)

out_H = tf.clip_by_value(x, 0, 1, name='out_H')
print(out_H.shape,"out_H.shape")
cost = Huber(y_true=H_out_true, y_pred=out_H, delta=0.01)

learning_rate = 0.001
learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name='learning_rate')
learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)

# total train epochs
num_epochs = 100

# Session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.compat.v1.train.Saver()


with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # tf.global_variables_initializer().run()
    for global_step in range(num_epochs):
        if global_step != 0 and np.mod(global_step, 10) == 0:
            sess.run(learning_rate_decay_op)
        total_train_loss = 0
        total_valid_loss = 0
        print("-------------------------- Epoch {:3d} ----------------------------".format(global_step))
        for i in range(5):
            print("---------- optimize sess.run start ----------")
            for j in range(X_dataset.shape[0]):
                in_L = X_dataset[j]  # select T_in frames
                in_L = in_L[np.newaxis, :, :, :, :]
                print(Y_dataset[j].shape, "asdasdasdads")

                sess.run(optimizer, feed_dict={H_out_true: Y_dataset[j][3], L: in_L, is_train: True})
                print("optimize:" + str(i) + " " + str(j) + " finished.")
        print("---------- train cost sess.run start -----------")
        for j in range(X_dataset.shape[0]):
            in_L = X_dataset[j]  # select T_in frames
            in_L = in_L[np.newaxis, :, :, :, :]
            print(Y_dataset[j].shape,"asdasdasdads")
            train_loss = sess.run(cost, feed_dict={H_out_true: Y_dataset[j][3], L: in_L, is_train: True})
            total_train_loss = total_train_loss + train_loss
            # print('this single train cost: {:.7f}'.format(train_loss))
            print("train cost :" + str(i) + " " + str(j) + " finished.")
        # for j in range(x_valid_data.shape[0]):
        #     in_L = x_valid_data_padded[j:j + T_in]  # select T_in frames
        #     in_L = in_L[np.newaxis, :, :, :, :]
        #     valid_loss = sess.run(cost, feed_dict={H_out_true: y_valid_data[j], L: in_L, is_train: True})
        #     total_valid_loss = total_valid_loss + valid_loss
        #     # print('this single valid cost: {:.7f}'.format(valid_loss))
        #     print("valid cost :" + str(i) + " " + str(j) + " finished.")
        # avg_train_loss = total_train_loss / X_dataset.shape[0]
        # avg_valid_loss = total_valid_loss / x_valid_data.shape[0]
        print("Epoch - {:2d}, avg loss on train set: {:.7f}, avg loss on valid set: {:.7f}.".format(global_step,
                                                                                                    123,
                                                                                                    123))
                                                                                                    # avg_valid_loss))
        if global_step == 0:
            with open('./logs/pb_graph_log.txt', 'w') as f:
                f.write(str(sess.graph_def))
            var_list = tf.compat.v1.global_variables()
            with open('./logs/global_variables_log.txt', 'w') as f:
                f.write(str(var_list))

        tf.compat.v1.train.write_graph(sess.graph_def, '.', './checkpoint/duf_' + str(global_step) + '.pbtxt')
        saver.save(sess, save_path="./checkpoint/duf", global_step=global_step)
        freeze_graph(check_point_folder='./checkpoint/', model_folder='./model',
                     pb_name='My_Duf_' + str(global_step) + '.pb')





