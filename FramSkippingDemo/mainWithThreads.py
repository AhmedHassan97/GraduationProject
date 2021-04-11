from FSM import *
from imports import *
from CDM import *

if __name__ == "__main__":

    start = time.time()
    AfterrSkipping = 0
    # Some Initialization
    rightPath = False
    while not rightPath:
        path = input("Enter the path of the video (ex: Test.mp4), To Exit Enter 'e' : ")
        if path == 'e':
            exit()
        if os.path.exists(path):
            rightPath = True
        else:
            print("Please Enter a righ Path, To Exit Enter 'e' ")

    # get bitrate
    # bitrate = GetBitRate(path)
    props = get_video_properties("" + path + "")
    bitrate = props['bit_rate']

    # print(bitrate, "Bitrate")
    # get number of frames
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    FramesPerIteration = 0
    if frameWidth * frameHeight <= 2073600:  # 1920 x 1080
        FramesPerIteration = 3000
    elif frameWidth * frameHeight <= 3686400:  # 2560 x 1440
        FramesPerIteration = 2000
    elif frameWidth * frameHeight <= 8294400:  # 3840 x 2160
        FramesPerIteration = 1000
    else:
        FramesPerIteration = 500

    print("Frames per Iteration:", FramesPerIteration)
    print("Total Number of frames of the video:", frameCount)

    # Get Video Duration
    video = VideoFileClip(path)
    video_duration = int(video.duration)
    print("Video duration in seconds:", video_duration)

    # calculate real Fps
    realFps = frameCount / video_duration
    print("video FPS:", realFps)

    iteration = math.ceil(float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / FramesPerIteration)
    try:
        os.remove('Skipped_Frames.txt')
    except:
        print("")
    try:
        os.makedirs("tempFolder")
    except:
        shutil.rmtree("tempFolder")
        os.makedirs("tempFolder")

    for i in range(iteration):
        rgbframes = GetFrames(path, 1, i * FramesPerIteration,
                              FramesPerIteration)  # take the filename, and the unit step

        firstCall = (len(rgbframes) / 5)
        results1 = []
        results2 = []
        results3 = []
        results4 = []
        results5 = []

        t1 = threading.Thread(target=CDMForThreads, args=(rgbframes[0:int(firstCall)], results1, FramesPerIteration, i, 1))
        t2 = threading.Thread(target=CDMForThreads,
                              args=(rgbframes[int(firstCall):int(2 * firstCall)],  results2,FramesPerIteration, i, 2))
        t3 = threading.Thread(target=CDMForThreads,
                              args=(rgbframes[int(2 * firstCall):int(3 * firstCall)],  results3,FramesPerIteration, i, 3))
        t4 = threading.Thread(target=CDMForThreads,
                              args=(rgbframes[int(3 * firstCall):int(4 * firstCall)],  results4,FramesPerIteration, i, 4))
        t5 = threading.Thread(target=CDMForThreads,
                              args=(rgbframes[int(4 * firstCall):int(5 * firstCall)], results5,FramesPerIteration, i, 5))
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

        AfterrSkipping += len(rgbFrames_final)
        convert_frames_to_video(rgbFrames_final, realFps, i)
        end = time.time()
    p1 = Process(target=fun, args=(int(iteration), bitrate, path))
    p1.start()
    p1.join()

    print("Number Of Frames After Skipping:", AfterrSkipping)
    # fun(int(iteration), bitrate, realFps)
    shutil.rmtree("tempFolder")
    os.remove("audio_from_video.mp3")

    print("Time:", str(round((end - start), 2)))
    # print(len(Gray_frames))
