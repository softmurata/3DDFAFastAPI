import sys
sys.path.append("./Model")

from typing import List

from fastapi import FastAPI
from fastapi import File
import boto3


from prediction import read_image, preprocess, predict

import uvicorn

app = FastAPI()



@app.post("/api/predict")
def predict_image(file: List[bytes]=File(...)):
    image_file, infer_file = file
    infer_type = infer_file.decode("utf-8")
    image = read_image(image_file)
    image = preprocess(image)
    ## infer_type = "landmark", "dense", "render", "render_depth", "pncc", "obj"
    preds = predict(image, infer_type=infer_type)

    # ToDo
    # upload files
    # create results
    s3_client = boto3.resource("s3")
    s3_client.Bucket("murata-torchserve-db").upload_file(Filename="test_obj.obj", Key="test_obj.obj")


    return preds


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")

