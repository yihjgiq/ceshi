import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# =======================
# 定义 CBAM 模块
# =======================
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力模块
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_att = self.ca(x)
        sp_att_input = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        sp_att = self.sa(sp_att_input)
        return x * ch_att * sp_att


# =======================
# 定义带注意力机制的 ResNet50 模型
# =======================
class ResNetAtt(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2, num_classes=5):
        super().__init__()
        # 初始化基础 ResNet50 模型（不加载预训练权重）
        self.base = resnet50(weights=None)

        # 插入注意力模块：注意，每个 layer 用 nn.Sequential 包装，
        # 原有的 ResNet50 的 layerX 成为了索引为 0 的模块，
        # Attention 模块挂载为索引为 1 的模块。
        self._insert_attention([CBAM(256), CBAM(512), CBAM(1024), CBAM(2048)])

        # 加载预训练权重
        if weights is not None:
            self._load_pretrained(weights)

        # 双输出头设计
        self.binary_head = nn.Sequential(
            nn.Conv2d(2048, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=True)  # 确保输出为 (720, 1280)
        )

        self.instance_head = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
            nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=True)  # 确保输出为 (720, 1280)
        )

        # 移除 ResNet 原始分类头
        self.base.fc = nn.Identity()

    def _insert_attention(self, modules):
        # 对于每个 layer，将原有的 layer 放在 index 0，将 attention 模块放在 index 1
        self.base.layer1 = nn.Sequential(self.base.layer1, modules[0])
        self.base.layer2 = nn.Sequential(self.base.layer2, modules[1])
        self.base.layer3 = nn.Sequential(self.base.layer3, modules[2])
        self.base.layer4 = nn.Sequential(self.base.layer4, modules[3])

    def _load_pretrained(self, weights):
        # 加载预训练权重，可以是字符串路径或者 torchvision 的 weights 对象
        if isinstance(weights, str):
            pretrained_dict = torch.load(weights, map_location='cpu')
            if 'state_dict' in pretrained_dict:  # 针对 checkpoint 格式
                pretrained_dict = pretrained_dict['state_dict']
        else:
            # torchvision 官方预训练权重
            pretrained_dict = weights.get_state_dict(progress=True)

        # 获取当前模型 base 部分的 state dict
        model_dict = self.base.state_dict()

        # 构造一个新的字典，用于存放映射后的预训练参数
        remapped_dict = {}
        for k, v in pretrained_dict.items():
            # 跳过 fc 层
            if k.startswith("fc"):
                continue

            new_key = k
            # 对于 ResNet 中的 layer 部分，官方权重 key 形如 "layerX.Y...."
            # 但在我们模型中，经过包装后，原有的 layer 被挂载在索引 0 下，
            # 因此新 key 应该变为 "layerX.0.Y...."
            if k.startswith("layer"):
                parts = k.split(".")
                # 在 parts[0] 为 layer1, layer2, 等部分后面插入 "0"
                new_key = parts[0] + ".0." + ".".join(parts[1:])

            # 如果映射后的 key存在，并且形状匹配，则收录
            if new_key in model_dict and v.shape == model_dict[new_key].shape:
                remapped_dict[new_key] = v

        print(f"Loaded {len(remapped_dict)}/{len(model_dict)} parameters into base.")
        # 更新基础模型的参数
        model_dict.update(remapped_dict)
        self.base.load_state_dict(model_dict, strict=False)

    def forward(self, x):
        # 特征提取过程：按照 ResNet50 的流程
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)  # 输出尺寸: [B,256,H/4,W/4]，其中仅 layer1[0] 的参数来自预训练权重
        x = self.base.layer2(x)  # 输出尺寸: [B,512,H/8,W/8]
        x = self.base.layer3(x)  # 输出尺寸: [B,1024,H/16,W/16]
        x = self.base.layer4(x)  # 输出尺寸: [B,2048,H/32,W/32]

        # 双任务输出
        binary_mask = self.binary_head(x)
        instance_map = self.instance_head(x)

        return binary_mask, instance_map  # 返回二值掩码和实例分割结果


# =======================
# 定义损失函数
# =======================
class LaneDetectionLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(LaneDetectionLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, binary_pred, binary_target, instance_pred, instance_target):
        binary_loss = self.bce_loss(binary_pred, binary_target)
        instance_loss = self.ce_loss(instance_pred, instance_target)
        return self.alpha * binary_loss + (1 - self.alpha) * instance_loss


# =======================
# 测试代码
# =======================
if __name__ == '__main__':
    model = ResNetAtt(num_classes=5)  # 假设车道线类别数为5
    dummy = torch.randn(2, 3, 720, 1280)
    binary, instance = model(dummy)
    print(f"Binary out mask shape: {binary.shape}")    # 应输出: torch.Size([2, 1, 720, 1280])
    print(f"Instance out map shape: {instance.shape}")  # 应输出: torch.Size([2, 5, 720, 1280])
