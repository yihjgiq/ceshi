import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class TusimpleDataset(Dataset):
    def __init__(self, root_dir, json_pattern="label_data_*.json", txt_file="train_val_gt.txt",
                 transform=None, target_size=(1280, 720), keep_ratio=True, max_samples=20):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.max_samples = max_samples

        self.annotations = self._load_annotations(json_pattern)
        self.ann_map = self._build_annotation_map(self.annotations)

        self.img_label_pairs = self._load_image_label_pairs(txt_file)

        if not self.img_label_pairs:
            raise ValueError("未加载任何图像标签对，请检查 txt_file 是否正确。")

        print(f"[数据集] 加载样本数: {len(self.img_label_pairs)}")

        self.original_size = self._get_original_image_size(self.img_label_pairs[0][0])
        self.pad_info = self._compute_padding_info()

    def __len__(self):
        return len(self.img_label_pairs)

    # def __getitem__(self, idx):
    #     img_path, label_path, class_labels = self.img_label_pairs[idx]

    #     image_bgr = cv2.imread(str(img_path))
    #     if image_bgr is None:
    #         raise ValueError(f"无法读取图像: {img_path}")
    #     image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #     original_image = image.copy()

    #     label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    #     if label is None:
    #          raise ValueError(f"无法读取标签图像: {label_path}")

    #     rel_path = os.path.relpath(img_path, self.root_dir).replace("\\", "/")
    #     ann = self.ann_map.get(rel_path)
    #     if ann is None:
    #         print(f"[DEBUG] annotation not found for {rel_path}")
    #         raise ValueError(f"找不到与 {rel_path} 匹配的 annotation 数据")
    #     #if idx < 3:
    #        #print(f"[DEBUG] ann sample for index {idx}:", ann)

    #     binary_mask, instance_mask = self._generate_masks(ann, image.shape[:2])
    #     print(f"👉 原始 binary_mask 非零像素数量: {np.count_nonzero(binary_mask)}")
    #     #print(f"重要binary_mask: {binary_mask}")  # 打印 binary_mask 的形状
         
    #     image = self._resize_with_padding(image)
    #     binary_mask = self._resize_with_padding(binary_mask, is_mask=True)
    #     instance_mask = self._resize_with_padding(instance_mask, is_mask=True)
        
    #     #print(f"binary_mask1.1: {binary_mask}")

    #     image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    #     binary_mask = torch.from_numpy((binary_mask > 0).astype(np.float32)).unsqueeze(0)

        
    #     #print(f"binary_mask1: {binary_mask}")  # 打印 binary_mask 的形状
        
    #     instance_mask = torch.from_numpy(instance_mask).long()

    #     filename = os.path.basename(str(img_path))
    #     return image, binary_mask, instance_mask, class_labels, original_image, filename
    def __getitem__(self, idx):
        img_path, label_path, class_labels = self.img_label_pairs[idx]

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"无法读取标签图像: {label_path}")

        rel_path = os.path.relpath(img_path, self.root_dir).replace("\\", "/")
        ann = self.ann_map.get(rel_path)
        if ann is None:
            print(f"[DEBUG] annotation not found for {rel_path}")
            raise ValueError(f"找不到与 {rel_path} 匹配的 annotation 数据")

           # -------------------- 生成 mask --------------------
        binary_mask, instance_mask = self._generate_masks(ann, image.shape[:2])
        #print(f"👉 原始 binary_mask 非零像素数量: {np.count_nonzero(binary_mask)}")

            # -------------------- Resize 前的 DEBUG --------------------
        #print(f"[DEBUG] 原始 binary_mask shape: {binary_mask.shape}, unique values: {np.unique(binary_mask)}")
        #print(f"[DEBUG] 原始 instance_mask: {instance_mask.shape}")
        #print(f"[DEBUG] 原始 binary_mask: {binary_mask.shape}")
          # -------------------- Resize 操作 --------------------
        image = self._resize_with_padding(image)
        binary_mask = self._resize_with_padding(binary_mask, is_mask=True)
        instance_mask = self._resize_with_padding(instance_mask, is_mask=True)

         # -------------------- Resize 后 DEBUG --------------------
        #print(f"👉 Resize 后 binary_mask 非零像素数量: {np.count_nonzero(binary_mask)}")
        #print(f"👉 Resize 后 binary_mask unique 值: {np.unique(binary_mask)}")

          # -------------------- 转 tensor --------------------
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # ✅ 添加标准化操作 (ImageNet mean/std)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
         # 👇 特别注意：这里加 threshold 再转 float 是避免非 0 数值太小被吞掉
        binary_mask_tensor = torch.from_numpy((binary_mask > 0).astype(np.float32)).unsqueeze(0)
        #print(f"👉 tensor binary_mask 非零像素数量: {(binary_mask_tensor != 0).sum().item()}")
        #print(f"[DEBUG] 原始 instance_mask: {instance_mask_tensor.shape}")
        #print(f"[DEBUG] 原始 binary_mask_tensor: {binary_mask_tensor.shape}")
        instance_mask_tensor = torch.from_numpy(instance_mask).long()

        filename = os.path.basename(str(img_path))

        binary_masks= binary_mask_tensor
        instance_masks= instance_mask_tensor
        #print(f"👉binary_masks 非零像素数量: {(binary_masks != 0).sum().item()}")

        return image, binary_masks, instance_masks, class_labels, original_image, filename


    def _load_annotations(self, json_pattern):
        json_files = list(self.root_dir.glob(json_pattern))
        if not json_files:
            raise FileNotFoundError(f"未找到匹配的标注文件: {json_pattern}")

        annotations = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    annotations.append(json.loads(line))
                    if self.max_samples and len(annotations) >= self.max_samples:
                        break
            if self.max_samples and len(annotations) >= self.max_samples:
                break

        if not annotations:
            raise ValueError("未加载任何有效的标注数据。")

        return annotations

    def _build_annotation_map(self, annotations):
        ann_map = {}
        for ann in annotations:
            key = os.path.normpath(ann['raw_file']).replace("\\", "/")
            ann_map[key] = ann
        return ann_map

    def _load_image_label_pairs(self, txt_file):
        pairs = []
        txt_path = self.root_dir / txt_file

        if not txt_path.exists():
            raise FileNotFoundError(f"标签文件 {txt_path} 不存在")

        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                img_rel = Path(parts[0].lstrip("/"))
                label_rel = Path(parts[1].lstrip("/"))
                class_labels = list(map(int, parts[2:])) if len(parts) > 2 else []

                img_path = (self.root_dir / img_rel).resolve()
                label_path = (self.root_dir / label_rel).resolve()

                if not img_path.exists() or not label_path.exists():
                    continue

                rel_path = os.path.relpath(img_path, self.root_dir).replace("\\", "/")
                if rel_path not in self.ann_map:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                binary_mask, _ = self._generate_masks(self.ann_map[rel_path], img.shape[:2])
                if binary_mask.sum() == 0:
                    print(f"[过滤] 空 binary_mask, 跳过图像: {img_path}")
                    continue

                pairs.append((img_path, label_path, class_labels))
                #print(f"[加载] 有效图像: {img_path}")

        return pairs

    def _get_original_image_size(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取初始图像尺寸: {img_path}")
        return img.shape[:2]  # H, W

    def _compute_padding_info(self):
        original_h, original_w = self.original_size
        target_w, target_h = self.target_size

        if self.keep_ratio:
            scale = min(target_w / original_w, target_h / original_h)
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            left_pad = (target_w - new_w) // 2
            top_pad = (target_h - new_h) // 2
            return (left_pad, top_pad, scale)
        else:
            scale_w = target_w / original_w
            scale_h = target_h / original_h
            return (0, 0, scale_w, scale_h)

    def _resize_with_padding(self, img, is_mask=False):
        target_w, target_h = self.target_size
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        if self.keep_ratio:
            left_pad, top_pad, scale = self.pad_info
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
            padded = cv2.copyMakeBorder(
                resized,
                top_pad,
                target_h - new_h - top_pad,
                left_pad,
                target_w - new_w - left_pad,
                cv2.BORDER_CONSTANT,
                value=0
            )
            return padded
        else:
            return cv2.resize(img, (target_w, target_h), interpolation=interp)


    def _generate_masks(self, ann, image_shape):
        h, w = image_shape
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        for lane_id, lane in enumerate(ann['lanes']):
            points = [(x, y) for x, y in zip(lane, ann['h_samples']) if x >= 0]
            #print(f"[DEBUG] Lane {lane_id} points: {points}")

            if len(points) < 2:
               #print(f"[DEBUG] 跳过 Lane {lane_id}，有效点数: {len(points)}")
               continue

            # 检查是否越界
            for x, y in points:
                if x >= w or y >= h:
                    print(f"[WARNING] 坐标 ({x}, {y}) 超出 mask 尺寸 ({w}, {h})")

            # 转换成 OpenCV 所需格式
            pts = np.array(points, np.int32).reshape((-1, 1, 2))

            # 画 binary mask（白色线）
            before = binary_mask.copy()
            cv2.polylines(binary_mask, [pts], isClosed=False, color=255, thickness=5)
            after = binary_mask
            #print(f"[DEBUG] Lane {lane_id} - binary_mask sum diff: {np.sum(after - before)}")

         # 画 instance mask（lane_id+1 编号线）
            cv2.polylines(instance_mask, [pts], isClosed=False, color=lane_id + 1, thickness=5)

         # 类型转换
        binary_mask = (binary_mask > 0).astype(np.float32)
        instance_mask = instance_mask.astype(np.int32)
        #print(f"可恶 binary_mask non-zero count: {np.count_nonzero(binary_mask)}")
        return binary_mask, instance_mask
