import os
from Yolov5_onnx import ONNX_YOLOV5
import cv2


def detect_results(img_path, model_path, device):
    results = []
    model = ONNX_YOLOV5(model_path, device)
    path = os.listdir(img_path)

    for idx,file in enumerate(path):
        single_result = []
        pre_image_path = os.path.join(img_path, file)

        img = cv2.imread(pre_image_path)

        objs = model.detect(img, img.shape)
        print("{}/{}:{},finished.".format(idx+1,len(path),pre_image_path))
        for id,box in enumerate(objs):
            single_result.append(dict(boxes=box[:4], scores = box[4], classes = box[5]))
        results.append(single_result)

    return results