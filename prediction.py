import sys
sys.path.append("./Model")

from Model.FaceBoxes import FaceBoxes
from Model.TDDFA import TDDFA
from Model.utils.functions import draw_landmarks
from Model.utils.render import render
from Model.utils.pncc import pncc
from Model.utils.depth import depth
from Model.utils.serialization import ser_to_obj, ser_to_ply

from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import yaml

import subprocess

onnx_flag = False


def load_model():
    # config = yaml.load(open("Model/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    config = yaml.load(open("/home/ubuntu/murata/Media2Cloud/FastAPIServer/3DDFAFastAPI/Model/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)

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

    return face_boxes, tddfa

face_boxes, tddfa = load_model()

        

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    
    return pil_image


def preprocess(image):
    
    return pil2cv(image)


def predict(image, infer_type):

    if infer_type == "face_boxe":

        boxes = face_boxes(image)

        person_results = boxes[0]

        bj = {}
        names = ["x1", "y1", "x2", "y2", "scores"]

        for idx, n in enumerate(names):
            bj[n] = float(person_results[idx])
            
        return bj
    
    else:
        filename = ""
        boxes = face_boxes(image)
        param_lst, roi_box_lst = tddfa(image, boxes)

        if infer_type == "landmark":
            filename = "test_{}.png".format(infer_type)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            draw_landmarks(image, ver_lst, wfp=filename, dense_flag=False)
        elif infer_type == "dense":
            filename = "test_{}.png".format(infer_type)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            draw_landmarks(image, ver_lst, wfp=filename, dense_flag=True)

        elif infer_type == "render":
            filename = "test_{}.png".format(infer_type)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            render(image, ver_lst, tddfa.tri, wfp=filename, alpha=0.6, show_flag=True)
        
        elif infer_type == "render_depth":
            filename = "test_{}.png".format(infer_type)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, wfp=filename, dense_flag=True)
            depth(image, ver_lst, tddfa.tri, wfp=filename, show_flag=True)

        elif infer_type == "pncc":
            filename = "test_{}.png".format(infer_type)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            pncc(image, ver_lst, tddfa.tri, wfp=filename, show_flag=True)

        elif infer_type == "obj":
            # obj
            filename = "test_{}.obj".format(infer_type)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            ser_to_obj(image, ver_lst, tddfa.tri, height=image.shape[0], wfp=filename)
        
        else:
            raise ValueError("Cannot find infer type")


        
        # upload files

        results = {"filename": filename}

        return results



    return {}


def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

