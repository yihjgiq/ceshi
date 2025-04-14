import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 自定义 Grad-CAM Target
class SegmentationTarget:
    def __init__(self, category, mask=None):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        if isinstance(model_output, tuple):
            model_output = model_output[1]  # 取 instance_map 分支
        target = model_output[:, self.category, :, :]
        return target.mean() if self.mask is None else (target * self.mask).sum()


def generate_gradcam(model, dataloader, device, epoch, save_dir="outputs/gradcam", max_batches=5):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True  # 确保梯度开启

    os.makedirs(save_dir, exist_ok=True)

    # 选择 target layer（ResNet 的最后一个 CBAM 后的卷积）
    #target_layers = [model.base.layer4[0][-1]]
    target_layers = [model.base.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    pbar = tqdm(enumerate(dataloader), total=min(max_batches, len(dataloader)), desc="Generating GradCAM")

    for idx, batch in pbar:
        if idx >= max_batches:
            break

        image = batch[0].to(device)  # image tensor
        original_image = batch[4][0].numpy().astype(np.uint8)  # [H, W, 3]
        original_image = original_image.astype(np.float32) / 255.0  # 归一化
        original_image = original_image[:, :, ::-1]  # BGR ➜ RGB

        filename = batch[5][0]

        target_channel = 0
        targets = [SegmentationTarget(category=target_channel)]

        # ✅ 不要加 no_grad，这里需要梯度！！
        grayscale_cam = cam(input_tensor=image, targets=targets)[0]

        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

        save_path = os.path.join(save_dir, f"ep{epoch}_{idx}_{filename}")
        cv2.imwrite(save_path, visualization)

    print(f"[GradCAM] Saved to {save_dir}")




# 可选：手动叠加热力图的函数
def overlay_gradcam_on_image(image_tensor, gradcam):
    image_np = TF.to_pil_image(image_tensor.cpu()).convert("RGB")
    image_np = np.array(image_np)

    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    if heatmap.shape[:2] != image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

    overlayed = heatmap * 0.4 + np.float32(image_np) / 255
    overlayed = overlayed / np.max(overlayed)
    overlayed = np.uint8(255 * overlayed)

    return overlayed