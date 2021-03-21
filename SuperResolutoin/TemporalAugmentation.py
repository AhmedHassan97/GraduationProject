from Imports import *


def convert_frames_to_video(rgb_frames, realFps, counter):
    NumberOfFrames = len(rgb_frames)
    print(NumberOfFrames, "Number of frames")
    # f = sf.SoundFile('audio_from_video.wav')
    # inSeconds = (len(f) / f.samplerate)
    # fps = float(NumberOfFrames) / inSeconds
    # realFps = realFps.split("/")
    # realFps = float(realFps[0]) / float(realFps[1])

    height, width, layers = rgb_frames[0].shape
    # out = cv2.VideoWriter('project1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), realFps, (width, height))

    out = cv2.VideoWriter("" + "tempFolder" + "/" + "outputVideo" + "" + str(counter) + "" + ".mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), realFps,
                          (width, height))
    for i in range(len(rgb_frames)):
        out.write(rgb_frames[i])
    out.release()
    # video3 = VideoFileClip(r"" + "project1.mp4" + "")
    # video3.write_videofile(r"" + "project.mp4" + "", audio="" + "audio_from_video.mp3" + "",
    #                        bitrate=(str(int(bitrate) - (0.05 * int(bitrate)))))


def GetFrames(fileName, TA, StartCount, FramesPerIteration):
    isNegative = False
    if TA < 0:
        TA = abs(TA)
        isNegative = True

    cap = cv2.VideoCapture(fileName)
    print(StartCount)
    cap.set(1, int(StartCount))

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bufsize = 0
    if frameCount - StartCount > FramesPerIteration:
        bufsize = FramesPerIteration
    else:
        bufsize = frameCount - StartCount
    buf = np.empty((bufsize, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while fc < bufsize and ret:
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    SelectedFrames = []

    count = 0

    if TA == 1:
        if isNegative:
            return buf[::-1]
        else:
            return buf
    else:
        for i in range(0, buf.shape[0], TA):
            SelectedFrames.append(buf[i])
            # io.imshow(SelectedFrames[count])
            # io.show()
            # count += 1
        # # cv2.waitKey(0)
        # print(buf.shape)
        print(len(SelectedFrames))
        if isNegative:
            return SelectedFrames[::-1]
        else:
            return SelectedFrames
