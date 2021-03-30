from TemporalAugmentation import *
from Imports import *
from DownSampling import *
from utils import *

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

try:
    os.makedirs("datasetFolder")
except:
    shutil.rmtree("datasetFolder")
    os.makedirs("datasetFolder")

TAList = [3, -3]
TA = random.choice(TAList)
print("TA=", TA)
rgbframes = []
allFrames = []

for i in range(iteration):
    print("here")
    rgbframes = GetFrames(path, TA, i * FramesPerIteration,
                          FramesPerIteration)  # take the filename, and the unit step)
    print(len(rgbframes),"len(rgbframes)")
    for g in range(len(rgbframes)):
        allFrames.append(rgbframes[g])
    # convert_frames_to_video(rgbframes, realFps, g)
    print(len(allFrames),"len(allFrames)")


h = gkern(13, 1.6)  # 13 and 1.6 for x4
# h = h[:, :, np.newaxis, np.newaxis].astype(np.float32)

for i in range(len(allFrames) - 1):
    mag_matrix=getMotionMatrix( allFrames[i],allFrames[i+1])
    i_pixel,j_pixel=get144OfMostMotion(mag_matrix)
    AfterGaussian_0 = convolve2D(allFrames[i][i_pixel:i_pixel + 144, j_pixel:j_pixel + 144, 0], h)
    AfterGaussian_1 = convolve2D(allFrames[i][i_pixel:i_pixel + 144, j_pixel:j_pixel + 144, 1], h)
    AfterGaussian_2 = convolve2D(allFrames[i][i_pixel:i_pixel + 144, j_pixel:j_pixel + 144, 2], h)

    newimage = np.zeros((132, 132, 3), dtype="uint8")

    newimage[..., 0] = AfterGaussian_0
    newimage[..., 1] = AfterGaussian_1
    newimage[..., 2] = AfterGaussian_2

    newimage = quarter_res_avg(newimage)

    cv2.imwrite("" + "datasetFolder" + "/" + "outputImage" + "" + str(i + 1) + "" + ".jpg",
                newimage)

cap.release()
cv2.destroyAllWindows()

end = time.time()

print("Time:", str(round((end - start), 2)))
# print(len(Gray_frames))
