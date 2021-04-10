from imports import *


def GetFrames(fileName, unitstep, StartCount, FramesPerIteration):
    cap = cv2.VideoCapture(fileName)
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

    if unitstep == 1:
        return buf
    else:
        for i in range(0, buf.shape[0], unitstep):
            SelectedFrames.append(buf[i])
            # io.imshow(SelectedFrames[count])
            # io.show()
            # count += 1
        # # cv2.waitKey(0)
        # print(buf.shape)
        return SelectedFrames
