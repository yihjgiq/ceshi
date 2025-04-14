# postprocess.py
import torch
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


def postprocess_lanes(instance_output, binary_pred, eps=0.5, min_samples=20, min_cluster_size=50, fit_error_thresh=None):
    if isinstance(instance_output, torch.Tensor):
        instance_output = instance_output.detach().cpu().numpy()
    if isinstance(binary_pred, torch.Tensor):
        binary_pred = binary_pred.detach().cpu().numpy()

    embedding_dim, H, W = instance_output.shape
    binary_pred = binary_pred.astype(bool)

    coords = np.argwhere(binary_pred)  # (N, 2) -> (y, x)
    if len(coords) == 0:
        return []

    embeddings = instance_output[:, coords[:, 0], coords[:, 1]].T  # (N, C)
    
    if len(coords) > 5000:
        indices = np.random.choice(len(coords), 5000, replace=False)
        coords = coords[indices]
        embeddings = embeddings[indices]

    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    except MemoryError:
        print("[ERROR] DBSCAN 内存不足，跳过此图像聚类")
        return []

    labels = clustering.labels_
    lanes = []

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # noise

        cluster_mask = labels == cluster_id
        cluster_coords = coords[cluster_mask]

        if len(cluster_coords) < min_cluster_size:
            continue

        xs = cluster_coords[:, 1]
        ys = cluster_coords[:, 0]

        if len(xs) >= 5:
            try:
                poly = np.polyfit(ys, xs, deg=2)
                y_vals = np.arange(160, 720, 10)  # TuSimple采样标准
                x_vals = np.polyval(poly, y_vals)

                # 防止无效拟合（NaN/Inf）
                if np.any(np.isnan(x_vals)) or np.any(np.isinf(x_vals)):
                    continue

                # 可选：拟合误差筛选
                if fit_error_thresh is not None:
                    pred_xs = np.polyval(poly, ys)
                    fit_error = np.mean((pred_xs - xs) ** 2)
                    if fit_error > fit_error_thresh:
                        continue

                lane = []
                for x, y in zip(x_vals, y_vals):
                    if 0 <= x < W:
                        lane.append((int(x), int(y)))
                if len(lane) > 0:
                    lanes.append(lane)
            except:
                continue

    return lanes


