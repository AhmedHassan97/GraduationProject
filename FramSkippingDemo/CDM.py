from imports import *


# def GetBitRate(filename):
#     cmnd = ['ffprobe', '-show_format', '-pretty', '-loglevel', 'quiet', filename]
#     p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     out, err = p.communicate()
#     out = out.decode()
#     out = out.split()
#     bitrate = []
#     for i in range(len(out)):
#         bitrate = re.findall(r'bit_rate=\d+\.\d+', out[i])
#         if bitrate:
#             BitrateUnit = out[i + 1]
#             break
#     bitrate = float(re.findall(r'\d+\.\d+', bitrate[0])[0])
#     if BitrateUnit == "Kbit/s":
#         bitrate *= 1000
#     elif BitrateUnit == "Mbit/s":
#         bitrate *= 1000000
#     elif BitrateUnit == "Gbit/s":
#         bitrate *= 1000000000
#     elif BitrateUnit == "Tbit/s":
#         bitrate *= 1000000000000
#     if err:
#         print("========= error ========")
#         print(err)
#     print("Get Bitrate Done")
#     return bitrate


def fun(loopCounter, bitrate, path):
    video = VideoFileClip(path)  # 2.
    audio = video.audio  # 3.
    audio.write_audiofile("audio_from_video.mp3")  # 4.
    clipList = []
    for i in range(loopCounter):
        clipList.append(VideoFileClip(r"" + 'tempFolder' + "/" + "outputVideo" + "" + str(i) + "" + ".mp4"))
    video3 = concatenate_videoclips([clipList[i] for i in range(
        loopCounter)])  # keeps these files open until process ends. Therefore, a new process is created just for this

    video3.write_videofile(r"" + "finaOutputVideo.mp4" + "", audio="" + "audio_from_video.mp3" + "",
                           bitrate=(str(int(bitrate) - (0.05 * int(bitrate)))))


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


# def CDM(rgbFrames, bitrate, realFps, dict, procNumber):
#     i = 1
#     toBeDeleted = []
#     while i < len(rgbFrames) - 1:
#         hist_i, bins_i = histogram(rgb2gray(rgbFrames[i]))
#         hist_iPlus1, bins_iiPlus1 = histogram(rgb2gray(rgbFrames[i + 1]))
#         hist_iMinus1, bins_iiMinus1 = histogram(rgb2gray(rgbFrames[i - 1]))
#
#         summationAfter = 0
#         summationBefore = 0
#         for k in range(0, 256, 1):
#             summationAfter += (abs(hist_i[k] - hist_iPlus1[k]))
#         for k in range(0, 256, 1):
#             summationBefore += (abs(hist_i[k] - hist_iMinus1[k]))
#
#         Di_iPlus1 = summationAfter / (sum(hist_i) + sum(hist_iPlus1))
#         Di_iMinus1 = summationBefore / (sum(hist_i) + sum(hist_iMinus1))
#
#         if Di_iPlus1 < 0.05 and Di_iMinus1 < 0.05:
#             toBeDeleted.append(i)
#             i += 2
#             # time.sleep(0.0)
#         else:
#             i += 1
#             # time.sleep(0.0)
#
#     # print(len(rgbFrames), "before")
#     # rgbFrames_final = []
#     # for i in range(len(rgbFrames)):
#     #     if i in toBeDeleted:
#     #         continue
#     #     else:
#     #         rgbFrames_final.append(rgbFrames[i])
#     # print(len(rgbFrames_final), "after")
#     print("done")
#     dict[procNumber] = toBeDeleted
#     # convert_frames_to_video(rgbFrames_final, bitrate, realFps)


def CDMForThreads(rgbFrames, rgbFrames_final,FramesPerIteration,iteration,ThreadNumber):
    i = 1
    toBeDeleted = []
    while i < len(rgbFrames) - 1:
        hist_i, bins_i = histogram(rgb2gray(rgbFrames[i]))
        hist_iPlus1, bins_iiPlus1 = histogram(rgb2gray(rgbFrames[i + 1]))
        hist_iMinus1, bins_iiMinus1 = histogram(rgb2gray(rgbFrames[i - 1]))

        summationAfter = 0
        summationBefore = 0
        for k in range(0, 256, 1):
            summationAfter += (abs(hist_i[k] - hist_iPlus1[k]))
        for k in range(0, 256, 1):
            summationBefore += (abs(hist_i[k] - hist_iMinus1[k]))

        # Di_iPlus1 = summationAfter / (sum(hist_i) + sum(hist_iPlus1))
        # Di_iMinus1 = summationBefore / (sum(hist_i) + sum(hist_iMinus1))

        Di_iPlus1 = summationAfter / (rgbFrames[i].shape[0] * rgbFrames[i].shape[1] * 2)
        Di_iMinus1 = summationBefore / (rgbFrames[i].shape[0] * rgbFrames[i].shape[1] * 2)

        if Di_iPlus1 < 0.3 and Di_iMinus1 < 0.3:
            toBeDeleted.append(i)
            i += 2
            time.sleep(0.0)
        else:
            i += 1
            time.sleep(0.0)

    print(len(rgbFrames), "before")
    # rgbFrames_final = []

    with open('Skipped_Frames.txt', 'a') as file:
        for i in range(len(rgbFrames)):
            if i in toBeDeleted:
                # index = iteration * FramesPerIteration + j * ThreadNumber
                file.write("%i\n" % (iteration * FramesPerIteration + len(rgbFrames) * (ThreadNumber-1) + i))

            else:
                rgbFrames_final.append(rgbFrames[i])
    print(len(rgbFrames_final), "after")
    print("done")
    # convert_frames_to_video(rgbFrames_final, bitrate, realFps)

