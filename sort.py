"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

# 计算两个边界框的交并比
def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """

    bb_gt = np.expand_dims(bb_gt, 0)  # 通过在指定位置插入新的轴来扩展数组形状
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

# 转换数据格式，从[x1,y1,x2,y2] 转换为[中心点x,中心点y,面积,宽高比] 且转换后的数据形状为（4,1）
def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

# 转换数据格式，从[中心点x,中心点y,面积,宽高比] 转换为[x1,y1,x2,y2] 且转换后的数据形状为（1,4）
def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

# 卡尔曼滤波的实现
# 你可以在任何含有不确定信息的动态系统中使用卡尔曼滤波, 对系统下一步的走向做出有根据的预测, 即使伴随着各种干扰, 卡尔曼滤波总是能指出真实发生的情况.
class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
    Updates the state vector with observed bbox.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return convert_x_to_bbox(self.kf.x)


# 将物体检测到的BBox和卡尔满滤波器预测的BBox进行匹配, 返回结果: 匹配成功的二维数组, 未匹配成功的物体检测BBox, 未匹配成功的跟踪BBox
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    # todo 这里应该是只有len(detections)==0,才会有min(iou_matrix.shape) == 0
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:  # 物体检测的BBox和跟踪的BBox能够一一匹配
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)  # 匈牙利算法实现最佳匹配
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []  # 没有匹配上的物体检测BBox放入unmatched_detections, 表示有新的物体进入画面来，后面要新增跟踪器来跟踪新物体
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []  # 没有匹配上的跟踪器放入unmatched_trackers, 表示之前的跟踪物体离开画面了, 要删除对应的跟踪器
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []  # 遍历match_indices, 将IOU值小于iou_threshold的匹配结果分别放入unmatched_detections和unmatched_trackers列表中
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# SORT算法包装类
class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    # update方法实现了SORT算法
    # 输入是当前帧中所有物体的检测BBox集合, 包括物体的score
    # 输出是当前帧的物体跟踪BBox集合, 包括物体跟踪的ID
    # 算法要求每帧必须调用一次, 即使该帧的所有物体的检测结果为空, 例如np.empty((0,5)). 注意：返回的对象个数和检测的对象个数可能不同
    def update(self, dets=np.empty((0, 5))):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        # 根据当前卡尔曼滤波跟踪器的个数，来创建二维矩阵trks。 行号为卡尔曼滤波跟踪器的标识，列向量为跟踪结果BBox和物体跟踪ID
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):  # 循环遍历卡尔曼滤波跟踪器列表
            pos = self.trackers[t].predict()[0]  # 用卡尔曼滤波跟踪器t 产生对应物体在当前帧中预测的BBox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):  # todo 这里
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # todo trks 存放了上一帧中被跟踪的所有物体在当前帧预测的BBox
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 将物体检测的BBox与卡尔曼滤波跟踪器预测的BBox进行匹配, 获得跟踪成功的物体矩阵,新增物体的矩阵, 离开画面的物体矩阵
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # 跟踪成功的物体与ID放入ret列表
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # 判断卡尔曼滤波跟踪器的time_since_update已经大于max_age, 从跟踪器列表中删除卡尔曼滤波跟踪器
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0  # 总时间
    total_frames = 0  # 总帧数
    colours = np.random.rand(32, 3)  # 仅仅用于display, BBox的颜色种类
    if display:  # 显示跟踪结果
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    ('
                'https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    '
                '$ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')  # data/train/*/det/det.txt
    for seq_dets_fn in glob.glob(pattern):  # 循环处理多个数据集,第一个seq_dets_fn值为data/train/TUD-Stadtmitte/det/det.txt,以此类推
        mot_tracker = Sort(max_age=args.max_age,  # todo max_age代表什么？min_hits代表什么？iou_threshold代表什么？
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # 创建跟踪器实例, create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')  # 对于第一个数据集TUD-Stadtmitte, 加载det文件后得到一个(951, 10)大小的二维数组
        seq = seq_dets_fn[pattern.find('*'):].split('/')[0]  # seq 为数据集名称 TUD-Stadtmitte

        with open('output/%s.txt' % seq, 'w') as out_file:
            print("Processing %s." % seq)
            for frame in range(int(seq_dets[:, 0].max())):  # seq_dets第一列的最大值,即该数据集的总帧数
                frame += 1  # 帧数从1开始计数
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]  # 取出第一列等于当前帧的所有检测物体的BBox，放入dets中。
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if display:  # 如果需要显示当前帧的跟踪结果，需要先把图像显示到屏幕上
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()  # 获取当前时间
                trackers = mot_tracker.update(dets)  # 将当前帧的所有物体的检测结果,传入mot_tracker实例的update方法中,获得对所有物体的跟踪计算结果BBox
                cycle_time = time.time() - start_time  # 计算处理时间
                total_time += cycle_time  # 累计到总处理时间中

                for d in trackers:  # 将SORT算法更新的所有跟踪结果输出到output文件中，同时如果display为True, 则将跟踪结果逐一画到当前帧，并展示到屏幕上
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")
