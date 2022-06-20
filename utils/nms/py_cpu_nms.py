# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将x中的元素从小到大排列，提取其对应的index(索引)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]  # 取置信度最大的（即第一个）框
        keep.append(i)  # 将其作为保留的框
        # 以下计算置信度最大的框（order[0]）与其它所有的框
        # （order[1:]，即第二到最后一个）框的IOU，以下都是以向量形式表示和计算
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 计算xmin的max,即overlap的xmin
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 计算ymin的max,即overlap的ymin
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 计算xmax的min,即overlap的xmax
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 计算ymax的min,即overlap的ymax

        w = np.maximum(0.0, xx2 - xx1 + 1)  # 计算overlap的width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # 计算overlap的hight
        inter = w * h  # 计算overlap的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算并，-inter是因为交集部分加了两次

        # np.where(condition)满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，
        # 输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
        inds = np.where(ovr <= thresh)[0]  # 本轮，order仅保留IOU不大于阈值的下标
        order = order[inds + 1]  # 删除IOU大于阈值的框，因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    return keep
