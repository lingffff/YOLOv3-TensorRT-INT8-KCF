#================================================================
#
#   文件名：kmeans_anchors.py
#   描述：使用kmeans聚类算法计算YOLOv3模型用到的锚框anchors
#   修改人：ling@stu.pku.edu.cn
#   修改时间：2020-12-30
#   修改内容：整合kmeans和anchors计算代码，添加注释
#
#================================================================
import os
import sys
import numpy as np
DATASET_PATH = "./"
CLUSTERS = 9
SIZE = (640, 360)
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

# 加载YOLO格式的标注数据
def load_dataset(path):
    jpegimages = os.path.join(path, 'images')
    if not os.path.exists(jpegimages):
        print('no JPEGImages folders, program abort')
        sys.exit(0)
    labels_txt = os.path.join(path, 'labels')
    if not os.path.exists(labels_txt):
        print('no labels folders, program abort')
        sys.exit(0)
    label_file = os.listdir(labels_txt)
    print('label count: {}'.format(len(label_file)))
    dataset = []
    for label in label_file:
        with open(os.path.join(labels_txt, label), 'r') as f:
            txt_content = f.readlines()
        for line in txt_content:
            line_split = line.split(' ')
            roi_with = float(line_split[len(line_split)-2])
            roi_height = float(line_split[len(line_split)-1])
            if roi_with == 0 or roi_height == 0:
                continue
            dataset.append([roi_with, roi_height])

    return np.array(dataset)

# 加载数据集标注
data = load_dataset(DATASET_PATH)
# 对标注框进行kmeans聚类
out = kmeans(data, k=CLUSTERS)
# 打印anchors的准确度
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
# 整理输出anchors
def return_product(elem):
    return elem[0] * elem[1]
anchors = []
for (x, y) in zip(out[:, 0] * SIZE[0], out[:, 1] * SIZE[1]):
    anchors.append((x, y))
anchors.sort(key=return_product)

print("Boxes:")
for anchor in anchors:
    print("{:.0f},{:.0f}, ".format(anchor[0], anchor[1]), end='')
print('')