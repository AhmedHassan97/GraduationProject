import os
from Imports import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Ahmed1")
# model=[]
stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
# Size of input temporal radius
T_in = 7
# Upscaling factor
R = 4

rightPath = False

while not rightPath:
    path = input("Enter the path of the video (ex: Test.mp4), To Exit Enter 'e' : ")
    if path == 'e':
        exit()
    if os.path.exists(path):
        rightPath = True
    else:
        print("Please Enter a righ Path, To Exit Enter 'e' ")

try:
    os.makedirs("tempFolder")
except:
    shutil.rmtree("tempFolder")
    os.makedirs("tempFolder")

# get number of frames
cap = cv2.VideoCapture(path)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get Video Duration
video = VideoFileClip(path)
video_duration = int(video.duration)
print("Video duration in seconds:", video_duration)

# calculate real Fps
realFps = frameCount / video_duration
print("video FPS:", realFps)

props = get_video_properties("" + path + "")
bitrate = props['bit_rate']

FramesPerIteration = 0
if frameWidth * frameHeight <= 2073600:  # 1920 x 1080
    FramesPerIteration = 100
elif frameWidth * frameHeight <= 3686400:  # 2560 x 1440
    FramesPerIteration = 100
elif frameWidth * frameHeight <= 8294400:  # 3840 x 2160
    FramesPerIteration = 100
else:
    FramesPerIteration = 100

iteration = math.ceil(float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / FramesPerIteration)
model = keras.models.load_model(r'D:\semestre9\gp\GraduationProject\SuperResolutoin\No7sModel3')
print("Ahmed2")
# count = 0

for i in range(iteration):
    rgbframes = GetFrames(path, i * FramesPerIteration,
                          FramesPerIteration)  # take the filename, and the unit step
    print(len(rgbframes), "debug 1")

    # x_train_All = np.lib.pad(rgbframes, pad_width=((T_in // 2, T_in // 2), (0, 0), (0, 0), (0, 0)), mode='constant')
    for1Video = []
    forAll = []
    for j in range(rgbframes.shape[0]-6):
        x_train = rgbframes[j:j + T_in]
        print(x_train.shape)
        x_train = x_train[np.newaxis, :, :, :, :]
        # count += 1
        image = model.predict(x_train)

        for1Video.append(np.around(image[0, 0]*255).astype(np.uint8))
    convert_frames_to_video(for1Video, realFps, i)
    forAll.extend(for1Video)

fun(iteration, bitrate, path)





