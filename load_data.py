import numpy as np
import pickle

from io_utils import smooth_moving_average


def load_srt_raw_newPre(timeLen, timeStep, fs, channel_norm, time_norm, label_type):
    n_channs = 30
    n_points = 7500
    data_len = fs * timeLen
    n_segs = int((n_points / fs - timeLen) / timeStep + 1)
    print('n_segs:', n_segs)

    # 生成模拟数据
    n_subs = 10  # 假设有10个样本
    n_vids = 28
    chn = 30
    fs = 250
    sec = 30
    data = np.random.randn(n_subs, n_vids, chn, fs * sec)  # 生成随机数据来模拟真实的数据

    # data shape :(sub, vid, chn, fs * sec)
    print('data loaded:', data.shape)

    # 选择不同的视频片段
    if label_type == 'cls2':
        vid_sel = list(range(12))
        vid_sel.extend(list(range(16, 28)))
        data = data[:, vid_sel, :, :]  # sub, vid, n_channs, n_points
        n_videos = 24
    else:
        n_videos = 28

    print('classification:', label_type)

    data = np.transpose(data, (0, 1, 3, 2)).reshape(n_subs, -1, n_channs)

    # 数据标准化
    if channel_norm:
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :], axis=0)) / np.std(data[i, :, :], axis=0)

    if time_norm:
        data = (data - np.tile(np.expand_dims(np.mean(data, axis=2), 2), (1, 1, data.shape[2]))) / np.tile(
            np.expand_dims(np.std(data, axis=2), 2), (1, 1, data.shape[2])
        )

    n_samples = np.ones(n_videos) * n_segs

    # 根据分类类型生成标签
    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1, 4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5, 9):
            label.extend([i] * 3)
        print(label)
    elif label_type == 'cls3':
        label = [0] * 12
        label.extend([1] * 4)
        label.extend([2] * 12)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]] * n_segs

    return data, label_repeat, n_samples, n_segs


def load_srt_pretrainFeat(datadir, channel_norm, timeLen, timeStep, isFilt, filtLen, label_type):
    if label_type == 'cls2':
        n_samples = np.ones(24).astype(np.int32) * 30
    else:
        n_samples = np.ones(28).astype(np.int32) * 30

    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    # 生成模拟数据
    data = np.random.randn(45, int(np.sum(n_samples)), 256)  # 假设有45个样本，每个样本有256个特征

    print(data.shape)
    print(np.min(data), np.median(data))

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))

    if isFilt:
        print('filtLen', filtLen)
        data = data.transpose(0, 2, 1)
        for i in range(data.shape[0]):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid + 1])] = smooth_moving_average(data[
                                                                                                         i, :,
                                                                                                         int(
                                                                                                             n_samples_cum[
                                                                                                                 vid]): int(
                                                                                                             n_samples_cum[
                                                                                                                 vid + 1])],
                                                                                                         filtLen)
        data = data.transpose(0, 2, 1)

    # 数据标准化
    if channel_norm:
        print('subtract mean and divided by var')
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :], axis=0)) / (np.std(data[i, :, :], axis=0) + 1e-3)

    # 根据分类类型生成标签
    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1, 4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5, 9):
            label.extend([i] * 3)
        print(label)
    elif label_type == 'cls3':
        label = [0] * 12
        label.extend([1] * 4)
        label.extend([2] * 12)
        print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]] * n_samples[i]

    return data, label_repeat, n_samples