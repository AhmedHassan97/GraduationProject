from Imports import *

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()




with tf.compat.v1.Session() as session:
    newdata=data[0:7,]

    newdata = newdata[np.newaxis, :, :, :, :]
    result = session.run(L, feed_dict={L: newdata})
    print("-------------------------")
    print(result.shape)

    for i in range(20):
        for j in range(7):
            io.imshow(data[j,])
            io.show()
            io.imshow(result[0,j,])
            io.show()

