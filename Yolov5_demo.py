import pdb
import cv2
import sys
import argparse
import os
sys.path.append("../")
# from demo_utils import build_input, show_result, save_result_as_pickle
from Yolov5_detect_api import detect_results


def parse_args():
    parser = argparse.ArgumentParser(description='detect images')
    parser.add_argument('--testfile', default="/home/yangyuhang583/project/Yolov5/yolov5/Yolo_Detect/onnx_test_image", help='img path to detect')
    # parser.add_argument('--model_path', default="/examples/model/yolov5s_person_without_precompile.rknn")

    parser.add_argument('--detector_model', default="/home/yangyuhang583/project/Yolov5/yolov5/weights/bike_detect.onnx",help='path to onnx model')
    #parser.add_argument('--extractor_model', default="../models/resnet50_reid.onnx")
    parser.add_argument('--device', default="0", help="cuda device 0 or 0,1,2... or cpu")
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--savepath', default="./output/")
    args = parser.parse_args()
    return args

# def detect_results(path):
#     results = []
#     for idx,file in enumerate(path):
#         single_result = []
#         pre_image_path = os.path.join(args.testfile, file)

#         img = cv2.imread(pre_image_path)

#         objs = model.detect(img, img.shape)
#         print("{}/{}:{},finished.".format(idx+1,len(path),pre_image_path))
#         for id,box in enumerate(objs):
#             single_result.append(dict(boxes=box[:4], scores = box[4], classes = box[5]))
#         results.append(single_result)

#     return results

if "__main__" == __name__:

    # 参数解析
    opt = parse_args()

    detect_box = detect_results(opt.testfile, opt.detector_model, opt.device)
    
    # # 保存结果
    # save_result_as_pickle(total_output, args.savepath)
     
    # # 可视化
    # show_result(total_output, args.savepath, line_thickness=3, bexec=True)
 

    # 资源销毁
    # hbpd.hbcv_person_detect_deinit()
