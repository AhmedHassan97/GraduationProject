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
import random
import scipy.ndimage.filters as fi
import tensorflow as tf
import argparse
from skimage.exposure import rescale_intensity
import tensorflow as tf
import glob
from PIL import Image
from tensorflow.python.framework import graph_util
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, BatchNormalization, Lambda, Concatenate, Conv3D, Reshape , Softmax, Conv2D, ReLU
import cv2 as cv