from FSM import *
from imports import *
from CDM import *


if __name__ == "__main__":

    start = time.time()
    # Some Initialization
    path = "COSTA RICA IN 4K 60fps HDR (ULTRA HD).mp4"
    FramesPerIteration = 3000

    # get bitrate
    # bitrate = GetBitRate(path)
    props = get_video_properties("" + path + "")
    bitrate = props['bit_rate']

    print(bitrate, "Bitrate")
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

    for i in range(iteration):
        rgbframes = GetFrames(path, 1, i * FramesPerIteration,
                              FramesPerIteration)  # take the filename, and the unit step
        print(len(rgbframes))
        print("GetFrames Done")

        firstCall = int(len(rgbframes) / 5)

        results1 = []
        results2 = []
        results3 = []
        results4 = []
        results5 = []

        t1 = threading.Thread(target=CDMForThreads, args=(rgbframes[0:firstCall], results1,))
        t2 = threading.Thread(target=CDMForThreads, args=(rgbframes[firstCall:2 * firstCall], results2,))
        t3 = threading.Thread(target=CDMForThreads, args=(rgbframes[2 * firstCall:3 * firstCall], results3,))
        t4 = threading.Thread(target=CDMForThreads, args=(rgbframes[3 * firstCall:4 * firstCall], results4,))
        t5 = threading.Thread(target=CDMForThreads, args=(rgbframes[4 * firstCall:5 * firstCall], results5,))
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()

        rgbFrames_final = []
        rgbFrames_final.extend(results1)
        rgbFrames_final.extend(results2)
        rgbFrames_final.extend(results3)
        rgbFrames_final.extend(results4)
        rgbFrames_final.extend(results5)

        convert_frames_to_video(rgbFrames_final, realFps, i)
        end = time.time()
    p1 = Process(target=fun, args=(int(iteration), bitrate, path))
    p1.start()
    p1.join()
    # fun(int(iteration), bitrate, realFps)
    shutil.rmtree("tempFolder")
    os.remove("audio_from_video.mp3")

    print("Time:", str(round((end - start), 2)))
    # print(len(Gray_frames))
