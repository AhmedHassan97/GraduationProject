from Imports import *
from utils import create_model, LoadDataSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = create_model()
model.summary()
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)
AdamOp = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=AdamOp, loss=tf.keras.losses.Huber(delta=0.001), metrics=['accuracy'])

# x_test_All, y_test_All = LoadDataSet(9)
# model = keras.models.load_model('D:\semestre9\gp\GraduationProject\SuperResolutoin\My Model21')

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.

# The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)


for epoch in range(0,100):
    print("here1")
    for video in range(9):
        print("here2")
        X_TRAIN = []
        Y_TRAIN = []
        x_train_All, y_train_All = LoadDataSet(video)
        for every7 in range(len(x_train_All)):
            x_train = x_train_All[every7]
            y_train = y_train_All[every7][0]
            x_train = x_train[np.newaxis, :, :, :, :]
            model.fit(x_train, y_train, epochs=epoch+1, initial_epoch=epoch)
    model.save('My Model' + str(epoch))

    # for i in range(len(x_test_All)):
    #     print("test")
    #     x_test = x_test_All[i]
    #     y_test = y_test_All[i][3]
    #     x_test = x_test[np.newaxis, :, :, :, :]
    #     model.evaluate(x_test, y_test)
