import os
import pdb
import cv2
import numpy as np
import torch
import torchvision
from utils import onnx_model_init, parse_configs, letterbox, sigmoid, filter_boxes, nms_boxes


class ONNX_YOLOV5(object):

    # 初始化
    def __init__(self, model_path, npu_id):
         
        self.sess, self.input_names, self.output_names = onnx_model_init(model_path)
        # 参数配置
        dirpath = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(dirpath, "yolov5_onnx_config.yaml")
        self.params = parse_configs(config_file)
        self.work_size = self.params['worksize']


    # 检测
    def detect(self, img, img_shape):

        # 1.预处理
        #worksize = self.params["worksize"]
        workimg, gain = self._preprocess(img, self.work_size)

        # 2.前向推理
        #pdb.set_trace()
        pred = self.sess.run(self.output_names, {self.input_names[0]: workimg})
        
        #pytorch自带方式后处理
        pred = torch.tensor(pred[0])
        objs = self._NMS(pred, rawImg_shape=img_shape)
        # 3. 自定义后处理
        #objs = self._postprocess(pred[1:], gain, img)
        return objs


    # 预处理
    def _preprocess(self, img, worksize):

        # 保持宽高比resize+padding
        workimg, gain = letterbox(img, worksize)

        # bgr2rgb
        workimg = cv2.cvtColor(workimg, cv2.COLOR_BGR2RGB)  #HWC
        workimg = workimg.transpose(2, 0, 1)  # to 3x416x416  CHW
        workimg = workimg[None]  #扩展一个batch维度  [1,3,416,416]

        #workimg = np.concatenate((workimg[..., ::2, ::2], workimg[..., 1::2, ::2], workimg[..., ::2, 1::2], workimg[..., 1::2, 1::2]), 1)
        workimg = workimg.astype(np.float32)
        workimg /= 255.0

        return workimg, gain


    # 后处理
    def _postprocess(self, pred, gain, img):
        
        src_h, src_w = img.shape[:2]
        boxes, classes, scores = [], [], []

        for t in range(len(pred)):

            # 结果sigmod化
            input0_data = sigmoid(pred[t][0])       # (3, 80, 80, 7)
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))   # (80, 80, 3, 7)
            grid_h, grid_w, channel_n, predict_n = input0_data.shape

            # 根据层选择anchor
            anchors = [self.params['anchors'][i] for i in self.params['masks'][t]]

            # 有框置信度
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)

            # 类别置信度
            box_class_probs = input0_data[..., 5:]

            # 左上角点
            box_xy = input0_data[..., :2]

            # 宽高
            box_wh = input0_data[..., 2:4]

            # 计算出预测框
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.params['worksize']  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的左上
            box = np.concatenate((box_xy, box_wh), axis=-1)

            # 过滤
            res = filter_boxes(box, box_confidence, box_class_probs, self.params['conf_thres'])
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])

        # 同类做nms
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, self.params["iou_thres"])
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        objs = list()

        if len(nboxes) < 1:
            return objs
        
        # 转原图
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))
            objs.append(dict(box=[x1, y1, x2, y2], score=score, label=cl))

        return objs
    

    def _NMS(self,prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=(), max_det=300, rawImg_shape=[720,1280,3]):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            #pdb.set_trace()
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]


            output[xi] = x[i]
            output = output[0]
            #pdb.set_trace()
            #Rescale回原图
            output[:, :4] = self.scale_coords(self.work_size, output[:, :4], rawImg_shape).round()

        return output

    def xywh2xyxy(self, x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords


    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
