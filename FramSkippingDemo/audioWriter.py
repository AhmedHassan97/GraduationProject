import os, sys, subprocess, shlex, re
from subprocess import call


def probe_file(filename):
    cmnd = ['ffprobe', '-show_format', '-pretty', '-loglevel', 'quiet', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode()
    out = out.split()
    print(out)
    for i in range(len(out)):
        bitrate = re.findall(r'bit_rate=\d+\.\d+', out[i])
        if bitrate:
            BitrateUnit = out[i + 1]
            break
    floatBitrate = float(re.findall(r'\d+\.\d+', bitrate[0])[0])
    print(BitrateUnit)
    if BitrateUnit == "Kbit/s":
        floatBitrate *= 1000
    elif BitrateUnit == "Mbit/s":
        floatBitrate *= 1000000
    elif BitrateUnit == "Gbit/s":
        floatBitrate *= 1000000000
    elif BitrateUnit == "Tbit/s":
        floatBitrate *= 1000000000000
    print(floatBitrate)
    if err:
        print("========= error ========")
        print(err)
    return floatBitrate


path = "Marvel Studios Avengers- Endgame - Official Trailer.mp4"
probe_file(path)
