# from FSM import *
# from imports import *
# from CDM import *
#
#
# if __name__ == "__main__":
#     start = time.time()
#
#     manager = Manager()
#     results = manager.dict()
#
#     path = "COSTA RICA IN 4K 60fps HDR (ULTRA HD).mp4"
#
#     rgbframes, bitrate, realFps = GetFrames(path,
#                                             1)  # take the filename, and the unit step
#     print(len(rgbframes))
#     print("GetFrames Done")
#
#     firstCall = int(len(rgbframes) / 5)
#
#     t1 = Process(target=CDM, args=(rgbframes[0:firstCall], bitrate, realFps, results,1))
#     t2 = Process(target=CDM, args=(rgbframes[firstCall:2 * firstCall], bitrate, realFps, results,2))
#     t3 = Process(target=CDM, args=(rgbframes[2 * firstCall:3 * firstCall], bitrate, realFps, results,3))
#     t4 = Process(target=CDM, args=(rgbframes[3 * firstCall:4 * firstCall], bitrate, realFps, results,4))
#     t5 = Process(target=CDM, args=(rgbframes[4 * firstCall:5 * firstCall], bitrate, realFps, results,5))
#     t1.start()
#     t2.start()
#     t3.start()
#     t4.start()
#     t5.start()
#     t1.join()
#     t2.join()
#     t3.join()
#     t4.join()
#     t5.join()
#
#     rgbFrames_final = []
#     for i in range(5):
#         array = rgbframes[i*firstCall:(i+1)*firstCall]
#         for j in range(len(array)):
#             print(len(results[i+1]))
#             if j in results[i+1]:
#                 continue
#             else:
#                 rgbFrames_final.append(array[j])
#
#     print(len(rgbFrames_final))
#     convert_frames_to_video(rgbFrames_final, bitrate, realFps)
#     end = time.time()
#     print("Time:", str(round((end - start), 2)))
#
