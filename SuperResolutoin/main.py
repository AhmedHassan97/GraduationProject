from TemporalAugmentation import *
from Imports import *

start = time.time()
# Some Initialization
path = "COSTA RICA IN 4K 60fps HDR (ULTRA HD).mp4"
FramesPerIteration = 3000

# get number of frames
cap = cv2.VideoCapture(path)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frameCount, "Number of frames")

# Get Video Duration
video = VideoFileClip(path)
video_duration = int(video.duration)
print(video_duration, "duration")

# calculate real Fps
realFps = frameCount / video_duration
print(realFps)

iteration = math.ceil(float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / FramesPerIteration)
try:
    os.makedirs("tempFolder")
except:
    shutil.rmtree("tempFolder")
    os.makedirs("tempFolder")

TAList = [1, 2, 3, -1, -2, -3]
TA = random.choice(TAList)
print("TA=", TA)
for i in range(iteration):
    rgbframes = GetFrames(path, TA, i * FramesPerIteration,
                          FramesPerIteration)  # take the filename, and the unit step
    convert_frames_to_video(rgbframes, realFps, i)

    print(len(rgbframes))
    print("GetFrames Done")

end = time.time()

print("Time:", str(round((end - start), 2)))
# print(len(Gray_frames))
