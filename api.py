import cv2
import yaml
import sys
sys.path.append("./Model")

from Model.FaceBoxes import FaceBoxes
from Model.TDDFA import TDDFA
from Model.utils.render import render
from Model.utils.pncc import pncc
from Model.utils.uv import uv_tex
from Model.utils.pose import viz_pose
from Model.utils.serialization import ser_to_ply, ser_to_obj
from Model.utils.functions import draw_landmarks, get_suffix

import matplotlib.pyplot as plt
from skimage import io

# load configuration
config = yaml.load(open("Model/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
onnx_flag = True

if onnx_flag:
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "4"

    from Model.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from Model.TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**config)
else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **config)


# read image
img_url = "Model/examples/inputs/emma.jpg"
img = io.imread(img_url)
img = img[..., ::-1]  # RGB => BGR

# face detection
boxes = face_boxes(img)
print(boxes)
