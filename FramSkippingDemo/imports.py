import cv2
import numpy as np
from skimage.color import rgb2gray
import skimage.io as io
from skimage.exposure import histogram
import cv2
import os
from os.path import isfile, join
from moviepy.editor import *
import soundfile as sf
from videoprops import get_video_properties  #To get bitrate
import shutil #To delete folders
import threading
import time
from multiprocessing import Process
import concurrent.futures
from multiprocessing import *
import os, sys, subprocess, shlex, re
from subprocess import call
import math