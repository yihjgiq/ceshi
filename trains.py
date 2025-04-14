import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ColorJitter, RandomHorizontalFlip
from torchvision.models import ResNet50_Weights
from sklearn.metrics import f1_score, accuracy_score
from models.resnet_att import ResNetAtt
from dataset.TusimpleDataset import TusimpleDataset
from gradcam_utils import generate_gradcam
from postprocess import postprocess_lanes
import cv2
import random
import torch
import torch.nn as nn
import torchvision.utils as vutils
import json
import torchvision.transforms.functional as TF
from torchvision import transforms

class LaneDetectionLoss(nn.Module):
    def __init__(self, alpha=1, delta_v=0.5, delta_d=1.5, embedding_weight=1.0, device='cpu', warmup_epochs=5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0).to(device))
        self.alpha = alpha  # BCE loss权重
        self.delta_v = delta_v  # 同类embedding距离容忍阈值
        self.delta_d = delta_d  # 不同类embedding最小间距
        self.embedding_weight = embedding_weight
        self.device = device
        self.warmup_epochs = warmup_epochs  # 默认暖启动轮次

    def forward(self, preds, targets, epoch=None):
        binary_pred, instance_pred = preds
        binary_target, instance_target = targets

        if binary_target.ndim == 3:
            binary_target = binary_target.unsqueeze(1)

        # Binary BCE Loss
        binary_loss = self.bce_loss(binary_pred, binary_target.float())

        # Warmup阶段不计算embedding loss
        if epoch is not None and epoch < self.warmup_epochs:
            loss_embed = torch.tensor(0.0, device=self.device)
        else:
            loss_embed = self.embedding_loss(instance_pred, instance_target)

        if epoch is not None:
            print(f"[DEBUG] Epoch {epoch} - Loss BCE: {binary_loss.item():.4f}, Embed: {loss_embed.item():.4f}")
            print(f"[DEBUG] Binary target sum: {binary_target.sum().item()}")  # 若为 0，说明全黑

        return self.alpha * binary_loss + self.embedding_weight * loss_embed

    def embedding_loss(self, embeddings, instance_masks):
        batch_size, embedding_dim, H, W = embeddings.size()
        total_var_loss = 0.0
        total_dist_loss = 0.0
        total_inst = 0

        for b in range(batch_size):
            embedding = embeddings[b].view(embedding_dim, -1)  # [C, H*W]
            instance_mask = instance_masks[b].view(-1)  # [H*W]

            unique_ids = instance_mask.unique()
            unique_ids = unique_ids[unique_ids != 0]  # remove background

            if len(unique_ids) < 1:
                continue

            means = []
            var_loss = 0.0

            for inst_id in unique_ids:
                mask = (instance_mask == inst_id)
                if mask.sum() < 5:
                    continue

                embed_i = embedding[:, mask]  # [C, N]
                mean_i = embed_i.mean(dim=1, keepdim=True)  # [C, 1]
                means.append(mean_i)

                dist = (embed_i - mean_i).norm(p=2, dim=0)
                var = torch.clamp(dist - self.delta_v, min=0.0) ** 2
                var_loss += var.mean()
                total_inst += 1

            if len(means) > 1:
                means = torch.stack(means)  # [num_inst, C, 1]
                num_pairs = 0
                dist_loss = 0.0
                for i in range(len(means)):
                    for j in range(i + 1, len(means)):
                        dist = (means[i] - means[j]).norm()
                        dist_loss += torch.clamp(2 * self.delta_d - dist, min=0.0) ** 2
                        num_pairs += 1
                dist_loss /= max(1, num_pairs)
            else:
                dist_loss = torch.tensor(0.0, device=embeddings.device)

            total_var_loss += var_loss
            total_dist_loss += dist_loss

        # Normalize by batch or total_inst
        if total_inst > 0:
            total_var_loss /= total_inst
        total_loss = total_var_loss + total_dist_loss
        return total_loss / batch_size




def evaluate(model, val_loader, device):
    model.eval()
    y_true, y_pred = [], []
    
    debug_saved = False  # 标志只保存一次
    os.makedirs("debug_val", exist_ok=True)

    with torch.no_grad():
        for images, binary_masks, _, _, _, _ in val_loader:
            images = images.to(device)
            binary_masks = binary_masks.to(device)
            #print(f"👉  binary_masks2 非零像素数量: {(binary_masks != 0).sum().item()}")
            #print(f"👉binary_mask: {binary_masks.shape}")
            # 打印binary_masks的内容和最小最大值
            #print(f"binary_masks: {binary_masks}")
            #print(f"binary_masks min: {binary_masks.min().item()}, max: {binary_masks.max().item()}")  # Debug info
            
            preds, _ = model(images)
            probs = torch.sigmoid(preds)
            
            
            if probs.dim() == 4 and probs.shape[1] == 1:
                probs = probs.squeeze(1)  # [B, H, W]

            
            preds_bin = (probs > 0.5).long()

            # 打印预测结果和正类像素数量
            #print(f"preds_bin: {preds_bin}")
            #print(f"Predicted positive pixels: {(preds_bin > 0.5).sum().item()}")

            print(f"[DEBUG] probs mean: {probs.mean().item():.4f}, max: {probs.max().item():.4f}, min: {probs.min().item():.4f}")
            print(f"[DEBUG] GT positive pixels: {(binary_masks > 0.5).sum().item()}")
            print(f"[DEBUG] pred positive pixels: {(preds_bin  > 0.7).sum().item()}")


            # Make sure preds and masks are both [B, H, W]
            if probs.dim() == 4 and probs.shape[1] == 1:
                probs = probs.squeeze(1)  # [B, H, W]

            preds_bin = (probs > 0.5).long()
            
            if not debug_saved:
                vutils.save_image(binary_masks.float(), f"debug_val/val_gt_epoch0.png")
                vutils.save_image(probs.unsqueeze(1), f"debug_val/val_pred_epoch0.png")  # 注意：需要 [B, 1, H, W]，加回一个维度
                debug_saved = True

            # Flatten for sklearn
            y_true.extend(binary_masks.cpu().numpy().astype(np.uint8).reshape(-1))
            y_pred.extend(preds_bin.cpu().numpy().astype(np.uint8).reshape(-1))
        
        #print(f"True labels: {y_true}")
        #print(f"Predicted labels: {y_pred}")

    # Now y_true and y_pred are both 1D arrays
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, f1



def draw_lanes(image, fitted_lanes, color=None, thickness=2):
    img = image.copy()
    for lane in fitted_lanes:
        lane_color = color or [random.randint(100, 255) for _ in range(3)]
        for i in range(1, len(lane)):
            pt1 = tuple(map(int, lane[i - 1]))
            pt2 = tuple(map(int, lane[i]))
            cv2.line(img, pt1, pt2, lane_color, thickness)
    return img

def convert_to_tusimple_format(fitted_lanes, h_samples, img_path):
    result = {
        "lanes": [],
        "h_samples": h_samples,
        "raw_file": img_path
    }
    for lane in fitted_lanes:
        x_coords = []
        for h in h_samples:
            x_at_h = [int(x) for x, y in lane if int(y) == h]
            x_coords.append(x_at_h[0] if x_at_h else -2)
        result["lanes"].append(x_coords)
    return result



def visualize_prediction(model, val_loader, device, epoch, output_dir, save_dir=None, write_json=True):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_results = []
    h_samples = list(range(160, 720, 10))  # TuSimple 标准 y 坐标

    with torch.no_grad():
        for idx, (images, binary_masks, _, _, _, filenames) in enumerate(val_loader):
            image = images[0].to(device).unsqueeze(0)  # shape: [1, 3, H, W]
            filename = filenames[0]

            # --- Prepare data for visualization ---
            input_img = images[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3], for imshow
            input_img = (input_img * 255).astype(np.uint8)        # denormalize if needed

            gt_mask = binary_masks[0].squeeze().cpu().numpy()     # [H, W]
            binary_output, instance_output = model(image)

            print("[DEBUG] binary_output shape:", binary_output.shape)
            print("[DEBUG] binary_output logits:", binary_output.min().item(), binary_output.max().item(), binary_output.mean().item())

            # --- Prediction ---
            pred_mask = torch.sigmoid(binary_output).squeeze().cpu().numpy()  # [H, W]
            pred_bin = (pred_mask > 0.5).astype(np.uint8)

            # --- Lane拟合 ---
            fitted_lanes = postprocess_lanes(instance_output[0], pred_bin)

            # --- 可视化 ---
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(input_img)
            axs[0].set_title("Input Image")
            axs[1].imshow(gt_mask, cmap='gray')
            axs[1].set_title("GT Binary Mask")
            axs[2].imshow(pred_bin, cmap='gray')
            axs[2].set_title("Predicted Binary Mask")
            axs[3].imshow(input_img)
            if isinstance(fitted_lanes, list):
                for lane in fitted_lanes:
                    axs[3].plot([x for x, y in lane], [y for x, y in lane], color='lime', linewidth=2)
            axs[3].set_title("Fitted Lane Curves")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/epoch{epoch}_vis_{os.path.basename(filename)}.png")
            plt.close()

            # --- Binary heatmap ---
            plt.imshow(pred_mask, cmap='hot')
            plt.title("Binary Output Heatmap (sigmoid)")
            plt.colorbar()
            plt.savefig(f"{output_dir}/epoch{epoch}_binary_heatmap_{os.path.basename(filename)}.png")
            plt.close()

            # --- TuSimple JSON result ---
            if write_json:
                json_result = convert_to_tusimple_format(fitted_lanes, h_samples, filename)
                all_results.append(json_result)

            if idx == 9:  # 控制最多输出10张
                break

    if write_json:
        with open(f"{output_dir}/epoch{epoch}_tusimple_predictions.json", "w") as f:
            json.dump(all_results, f, indent=2)



# 生成 Grad-CAM 可视化
# def debug_visualize_val_sample(val_set):
#     import matplotlib.pyplot as plt
#     import torchvision.transforms.functional as TF

#     idx = random.randint(0, len(val_set)-1)
#     image, binary_mask, _, _, _, filename = val_set[idx]

#     #print(f"[DEBUG] Visualizing file: {filename}")

#     fig, axs = plt.subplots(1, 2, figsize=(10, 4))
#     axs[0].imshow(TF.to_pil_image(image))
#     axs[0].set_title("Input Image")
#     axs[1].imshow(binary_mask.squeeze(), cmap='gray')
#     axs[1].set_title("GT Binary Mask")
#     for ax in axs:
#         ax.axis("off")
#     plt.tight_layout()
#     plt.show()



def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    with open("configs/tusimple.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetAtt(weights=ResNet50_Weights.IMAGENET1K_V2, num_classes=5).to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    transform = Compose([
        ColorJitter(0.3, 0.3, 0.3, 0.1),
        RandomHorizontalFlip(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Train dataset setup
    train_params = cfg['datasets']['train']['parameters']
    train_dataset = TusimpleDataset(
        root_dir=train_params['root'],
        json_pattern=train_params['json_pattern'],
        txt_file="seg_label/list/train_val_gt.txt",  # 或写成 train_params['txt_file']
        transform=transform,
        target_size=tuple(train_params['img_size']),
        keep_ratio=train_params.get('keep_ratio', True)
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Validation dataset setup
    val_params = cfg['datasets']['val']['parameters']
    val_dataset = TusimpleDataset(
        root_dir=val_params['root'],
        json_pattern=val_params['json_pattern'],  # 可选，没有就传空
        txt_file=val_params['gt_file'],  # 你配置里就是 train_val_gt.txt
        transform=transform,
        target_size=tuple(val_params['img_size']),
        keep_ratio=val_params.get('keep_ratio', True)
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=True)

    criterion = LaneDetectionLoss(alpha=1,  device=device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    for images, binary_mask, _, _, _, filename in train_loader:
        print("Image shape:", images.shape)
        print("Binary mask shape:", binary_mask.shape)
        print("Mask unique values:", torch.unique(binary_mask))
        break
    # Training Loop
    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for images, binary_masks, instance_masks, _, _, _ in loop:
            images, binary_masks, instance_masks = images.to(device), binary_masks.to(device), instance_masks.to(device)

            #print(f"Model input Instance mask shape: {instance_masks.shape}")
            if binary_masks.ndimension() == 3:  # 如果是 [B, H, W] 的形状
                binary_masks = binary_masks.unsqueeze(1)  # 将其转换为 [B, 1, H, W]

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, (binary_masks, instance_masks), epoch=epoch+1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation and Evaluation
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        avg_loss = total_loss / len(train_loader)
        val_loss = 1 - val_f1
        scheduler.step(val_loss)

        # Visualization after each epoch
        visualize_prediction(model, val_loader, device, epoch+1, "D:/project/New/output/vis")
        generate_gradcam(model, val_loader, device, epoch+1, save_dir="D:/project/New/output/gradcam")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #torch.save(model.state_dict(), "outputs/best_model.pth")
            torch.save(model.state_dict(), "D:/project/New/output/best_model.pth")
            early_stop_counter = 0
            print("[INFO] Best model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print("[EARLY STOP] No improvement.")
                break

    print("Training complete.")



if __name__ == '__main__':
    main()

