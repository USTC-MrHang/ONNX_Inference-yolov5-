#coding=utf-8
import pdb
import cv2
import yaml
import numpy as np
import torch



# def rknn_model_init(model_path, npu_id):

#     from rknn.api import RKNN
#     # 1. 创建对象
#     rknn = RKNN(verbose=False)

#     # 2. 获取设备号, 后改成参数指定
#     devs = rknn.list_devices()
#     device_id_dict = {}
#     for index, dev_id in enumerate(devs[-1]):
#         if dev_id[:2] != 'TS':
#             device_id_dict[0] = dev_id
#         if dev_id[:2] == 'TS':
#             device_id_dict[1] = dev_id
    
#     # 3. 加载模型
#     print('-->loading model : ' + model_path)
#     rknn.load_rknn(model_path)
#     # 初始化rknn环境
#     # print('--> Init runtime environment on: ' + device_id_dict[npu_id])
#     # ret = rknn.init_runtime(device_id=device_id_dict[npu_id])
#     ret = rknn.init_runtime(device_id=None)
#     if ret != 0:
#         print('Init runtime environment failed')
#         exit(ret)
#     print('done')
    
#     return rknn


def onnx_model_init(model_path):
    import onnxruntime
    # session = onnxruntime.InferenceSession(model_path)
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider']) # onnxruntime-gpu
    input_names = list(map(lambda x: x.name, session.get_inputs()))
    output_names = list(map(lambda x: x.name, session.get_outputs()))
    return session, input_names, output_names


# 解析配置文件
def parse_configs(config_file):
    with open(config_file, "rb") as f:
        params = yaml.load(f, yaml.FullLoader)
    return params


# 等宽高比
def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    # 等宽高比缩放
    new_img, scale = auto_resize(img, *new_wh)
    # pading到worksize
    shape = new_img.shape
    new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - shape[0], 0, new_wh[0] - shape[1], cv2.BORDER_CONSTANT, value=color)
    return new_img, (new_wh[0] / scale, new_wh[1] / scale)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thres):

    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


# 构建输出
def build_output(input_param, objs):

    output = dict()

    # 相机号
    if "equipmentId" in input_param:
        output["equipmentId"] = input_param["equipmentId"]

    # 透传字段
    if "traceId" in input_param:
        output["traceId"] = input_param["traceId"]

    # 时间戳
    if "timeStamp" in input_param:
        output["timeStamp"] = input_param["timeStamp"]

    # 详情结果
    output["detail"] = dict()

    # 结果
    output["result"] = objs


    return output




# 检测框格式转换
def fcos_xyxy2xywh(bbox):

    xmin, ymin, xmax, ymax = bbox.split(1, dim=-1)
    TO_REMOVE = 1
    bbox = torch.cat(
        (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
    )

    return bbox


# 类别框颜色
def compute_colors_for_labels(labels):

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    return colors


#TODO:将画框和画类别合并
# 绘制检测框
def overlay_boxes(image, prediction, categories):

    labels = prediction['labels']
    scores = prediction['scores']
    boxes = prediction['bbox']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 2)

    classes = [categories[i] for i in labels.tolist()]
    template = "{}: {:.2f}"
    for b, s, c in zip(boxes.tolist(), scores.tolist(), classes):
        x, y = b[:2]
        text = template.format(c, s)
        cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    return image


# 检测结果保存为评测格式
def eval_pkl(imgname, prediction):

    det = dict()
    det['traceId'] = imgname
    det['result'] = list()

    objs = prediction['bbox'].tolist()

    if objs:

        for i, obj in enumerate(objs):

            tmp=dict()
            tmp['box'] = obj
            tmp['score'] = prediction['scores'].tolist()[i]
            tmp['label'] = int(prediction['labels'].tolist()[i]-1) # 评测脚本类别从0开始
            det['result'].append(tmp)

    return det


# 非极大值抑制
def fcos_nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


# 等宽高比缩放
def get_size(image_size, worksize):
    w, h = image_size # image_size是原始图像尺寸, eg:(1280, 720)
    size = worksize[0]
    max_size = worksize[1] 
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        # 保持原始图像的宽高比，若resize后，长边>max_size，则按最长边计算宽高比获得短边size
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    # 有一个边与resize后边的size相同，直接返回原图大小
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    # 保持相同的宽高比
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1):

    # boxlist: dict()
    if nms_thresh <= 0 or len(boxlist['bbox']) < 1:
        return boxlist

    boxes = boxlist['bbox'].numpy()
    scores = boxlist['scores'].numpy()
    labels = boxlist['labels'].numpy()
    
    # 同类做nms
    nboxes, nlabels, nscores = [], [], []
    for i in set(labels):

        inds = np.where(labels == i)

        b = boxes[inds]
        l = labels[inds]
        s = scores[inds]

        keep = fcos_nms_boxes(b, s, nms_thresh)
        if max_proposals > 0:
            keep = keep[: max_proposals]

        nboxes.append(torch.from_numpy(b[keep]))
        nlabels.append(torch.from_numpy(l[keep]))
        nscores.append(torch.from_numpy(s[keep]))
    

    boxlist['bbox'] = torch.cat(nboxes)
    boxlist['labels'] = torch.cat(nlabels)
    boxlist['scores'] = torch.cat(nscores)
    
    return boxlist


# 检测框边界保护
def clip_to_image(bbox, w, h, remove_empty=True):

    TO_REMOVE = 1
    bbox[:, 0].clamp_(min = 0, max = w - TO_REMOVE)
    bbox[:, 1].clamp_(min = 0, max = h - TO_REMOVE)
    bbox[:, 2].clamp_(min = 0, max = w - TO_REMOVE)
    bbox[:, 3].clamp_(min = 0, max = h - TO_REMOVE)


# 检测框尺寸过滤
def remove_small_boxes(detections, save_min_size):

    # Only keep boxes with both sides >= min_size

    xywh_boxes = fcos_xyxy2xywh(detections)
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= save_min_size) & (hs >= save_min_size)
    ).nonzero().squeeze(1)

    return detections[keep]


# 计算p3-p7特征层对应输入层上的位置
def compute_locations(features, fpn_strides):

    locations = []
    for level, feature in enumerate(features):
        h, w = feature.shape[-2:]
        locations_per_level = compute_locations_per_level(h, w, fpn_strides[level])
        locations.append(locations_per_level)

    return locations


def compute_locations_per_level(h, w, stride):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1) # (h, w)-> (h*w)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2 # torch.Size([h*w, 2])
    return locations


# 计算单个特征层上的检测结果
def forward_for_single_feature_map(locations, box_cls, box_regression, centerness, image_sizes):

    # TODO:参数配置
    pre_nms_thresh = 0.05
    cfg_pre_nms_top_n = 1000
    save_min_size = 0

    # numpy.ndarray to tensor
    box_cls = torch.from_numpy(box_cls)
    box_regression = torch.from_numpy(box_regression)
    centerness = torch.from_numpy(centerness)

    N, C, H, W = box_cls.shape
    # put in the same format as locations
    box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
    box_cls = box_cls.reshape(N, -1, C).sigmoid()
    box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
    box_regression = box_regression.reshape(N, -1, 4)
    centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
    centerness = centerness.reshape(N, -1).sigmoid()

    candidate_inds = box_cls > pre_nms_thresh
    # candidate_inds = candidate_inds 
    pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
    pre_nms_top_n = pre_nms_top_n.clamp(max=cfg_pre_nms_top_n)

    # multiply the classification scores with centerness scores
    box_cls = box_cls * centerness[:, :, None]

    results = []
    for i in range(N):
        per_box_cls = box_cls[i]
        per_candidate_inds = candidate_inds[i] # 每个位置是否存在各类别的置信度，(h*w, cls), 0不存在，1存在
        per_box_cls = per_box_cls[per_candidate_inds] # 取出对应位置的置信度

        per_candidate_nonzeros = per_candidate_inds.nonzero()
        per_box_loc = per_candidate_nonzeros[:, 0] # position
        per_class = per_candidate_nonzeros[:, 1] + 1 # class

        per_box_regression = box_regression[i]
        per_box_regression = per_box_regression[per_box_loc] # get box according to position with score
        per_locations = locations[per_box_loc]

        per_pre_nms_top_n = pre_nms_top_n[i]

        if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
            per_box_cls, top_k_indices = \
                per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            per_class = per_class[top_k_indices]
            per_box_regression = per_box_regression[top_k_indices]
            per_locations = per_locations[top_k_indices]

        # convert ltrb -> xyxy
        detections = torch.stack([
            per_locations[:, 0] - per_box_regression[:, 0],
            per_locations[:, 1] - per_box_regression[:, 1],
            per_locations[:, 0] + per_box_regression[:, 2],
            per_locations[:, 1] + per_box_regression[:, 3],
        ], dim=1)

        h, w = image_sizes[i]

        # Border protection, cut off the box in the pad area
        clip_to_image(detections, w, h, remove_empty=False) 

        # Only keep boxes with both sides >= min_size
        detections = remove_small_boxes(detections, save_min_size) 

        boxlist =dict()
        boxlist['bbox'] = detections # detections is list
        boxlist['image_size'] = (int(w), int(h))
        boxlist['mode'] = "xyxy"
        boxlist['labels'] = per_class
        boxlist['scores'] = torch.sqrt(per_box_cls)

        results.append(boxlist)

    return results

# 合并多个特征层上的检测结果
def cat_boxlist(bboxes):
    
    # bboxes: tuple( dict_1, .., dict_5 )

    size = bboxes[0]['image_size']

    mode = bboxes[0]['mode']

    cat_bboxes = torch.cat([bbox['bbox'] for bbox in bboxes], dim=0)
    cat_labels = torch.cat([bbox['labels'] for bbox in bboxes], dim=0)
    cat_scores = torch.cat([bbox['scores'] for bbox in bboxes], dim=0)

    # a dict for a image
    cat_boxlists = dict()
    cat_boxlists['bbox'] = cat_bboxes # cat_bboxes: tensor
    cat_boxlists['labels'] = cat_labels
    cat_boxlists['scores'] = cat_scores
    cat_boxlists['image_size'] = size
    cat_boxlists['mode'] = mode

    return cat_boxlists


# 检测框过滤
def select_over_all_levels(boxlists, nms_thresh=0.6):

    # # TODO:参数配置
    fpn_post_nms_top_n = 100

    # boxlists: list( dict(), ..., dict() )

    # number of batch_size
    num_images = len(boxlists) 
    results = []
    for i in range(num_images):
        result = boxlist_nms(boxlists[i], nms_thresh) 
        number_of_detections = len(result['bbox'])
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > fpn_post_nms_top_n > 0:
            cls_scores = result['scores']
            image_thresh, _ = torch.kthvalue(cls_scores, number_of_detections - fpn_post_nms_top_n + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result['bbox'][keep]
            result['labels'][keep]
            result['scores'][keep]

        results.append(result) # list( dict() )

    return results

# 转回原图
def convert_size(boxlists, ratio):

    # ratio = (h_ratio, w_ratio)

    for boxlist in boxlists:

        box = boxlist['bbox']
        xmin, ymin, xmax, ymax = box.split(1, dim=-1)
        scaled_xmin = xmin * ratio[1]
        scaled_xmax = xmax * ratio[1]
        scaled_ymin = ymin * ratio[0]
        scaled_ymax = ymax * ratio[0]
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        boxlist['bbox'] = scaled_box

    return boxlists


# 根据各类别置信度过滤
def filter_by_conf(boxlists, conf_thres_for_classes):

    conf_thres_for_classes = torch.tensor(conf_thres_for_classes)

    for boxlist in boxlists:

        # select_top_predictions according to scores
        scores = boxlist['scores']
        labels = boxlist['labels']

        # 根据给定的阈值保留
        thresholds = conf_thres_for_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)

        boxlist['bbox'] = boxlist['bbox'][keep]
        boxlist['labels'] = boxlist['labels'][keep]
        boxlist['scores'] = boxlist['scores'][keep]

        # 按置信度排序
        scores = boxlist['scores']
        _, idx = scores.sort(0, descending=True)
        boxlist['scores'] = boxlist['scores'][idx]


# fcos检测格式转换为yolov5检测格式
def det_fcos2yolov5(boxlists):

    # TODO:未用batch images，一次只算一张图
    boxlist = boxlists[0]

    # [[x,y,x,y], [x,y,x,y]]
    fcos_boxes = boxlist['bbox'].tolist()
    fcos_labels = boxlist['labels'].tolist()
    fcos_scores = boxlist['scores'].tolist()

    objs = list()
    for i, box in enumerate(fcos_boxes):
        obj = dict()
        obj['box'] = list(map(int, box))
        obj['label'] = fcos_labels[i] - 1 
        obj['score'] = fcos_scores[i]
        objs.append(obj)

    return objs

