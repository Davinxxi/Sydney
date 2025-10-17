import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from new_loss import SupCon_Loss, Weighted_SupConLoss, DynamicMargin_Loss, ArcCos_Loss, NegDot_Loss, MeanNegDot_Loss, NegWeighted_SupConLoss, NegDynamicMargin_Loss, NegArcCos_Loss
from mpl_toolkits.mplot3d import Axes3D

# theta: 0 ~ pi
num_points = 200
theta = np.linspace(0, np.pi, num_points)
cos_theta = np.cos(theta)

feat_dim = 128

def make_features_3d(theta_ap, theta_an):
    anchor = torch.zeros((1, feat_dim))
    anchor[0, 0] = 1.0
    positive = torch.zeros((1, feat_dim))
    positive[0, 0] = np.cos(theta_ap)
    positive[0, 1] = np.sqrt(np.clip(1 - np.cos(theta_ap) ** 2, 0, 1))
    negative = torch.zeros((1, feat_dim))
    negative[0, 0] = np.cos(theta_an)
    negative[0, 1] = np.sqrt(np.clip(1 - np.cos(theta_an) ** 2, 0, 1))
    # 정규화
    anchor = torch.nn.functional.normalize(anchor, dim=1)
    positive = torch.nn.functional.normalize(positive, dim=1)
    negative = torch.nn.functional.normalize(negative, dim=1)
    features = torch.cat([anchor, positive, negative], dim=0)
    labels = torch.tensor([0, 0, 1])
    return features, labels

loss_classes = [
    (SupCon_Loss, 'SupCon_Loss'),
    (Weighted_SupConLoss, 'Weighted_SupConLoss'),
    (DynamicMargin_Loss, 'DynamicMargin_Loss'),
    (ArcCos_Loss, 'ArcCos_Loss'),
    (NegDot_Loss, 'NegDot_Loss'),
    (MeanNegDot_Loss, 'MeanNegDot_Loss'),
    (NegWeighted_SupConLoss, 'NegWeighted_SupConLoss'),
    (NegDynamicMargin_Loss, 'NegDynamicMargin_Loss'),
    (NegArcCos_Loss, 'NegArcCos_Loss'),
]


# -------------------------------
# 3D surface plot: feature angle (임베딩 각도), label angle (정답 각도), loss
# -------------------------------
num_points_3d = 50
feature_angles = np.linspace(0, np.pi, num_points_3d)  # feature(임베딩) 각도
label_angles = np.linspace(0, np.pi, num_points_3d)    # label(정답) 각도
FeatureGrid, LabelGrid = np.meshgrid(feature_angles, label_angles)

for loss_cls, name in loss_classes:
    loss_fn = loss_cls()
    Loss_surface = np.zeros_like(FeatureGrid)
    for i in range(num_points_3d):
        for j in range(num_points_3d):
            # feature_angle: anchor와의 각도, label_angle: 정답 각도
            feature_angle = FeatureGrid[j, i]
            label_angle = LabelGrid[j, i]
            # feature 벡터 생성 (anchor: [1,0,...], feature: [cos(feature_angle), sin(feature_angle), 0, ...])
            anchor = torch.zeros((1, feat_dim))
            anchor[0, 0] = 1.0
            feature = torch.zeros((1, feat_dim))
            feature[0, 0] = np.cos(feature_angle)
            feature[0, 1] = np.sqrt(np.clip(1 - np.cos(feature_angle) ** 2, 0, 1))
            # 정규화
            anchor = torch.nn.functional.normalize(anchor, dim=1)
            feature = torch.nn.functional.normalize(feature, dim=1)
            features = torch.cat([anchor, feature], dim=0)
            # label: [0(anchor), label_angle(정답)]
            # label은 각도(degree)로 저장 (0~180)
            # labels = torch.tensor([0, label_angle * 180 / np.pi])
            labels = torch.tensor([0, int(label_angle)])
            try:
                loss = loss_fn(features, labels, sigma=0).item()
            except Exception:
                loss = np.nan
            Loss_surface[j, i] = loss
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(FeatureGrid * 180 / np.pi, LabelGrid * 180 / np.pi, Loss_surface, cmap='viridis')

    # ax.set_xlim(0, 20)   # Feature Angle 축을 0~20도로 제한
    # ax.set_ylim(0, 20)   # Label Angle 축을 0~20도로 제한

    ax.set_xlabel('Feature Angle (deg)')
    ax.set_ylabel('Label Angle (deg)')
    ax.set_zlabel('Loss')
    ax.set_title(f'{name} 3D Loss Surface (Feature vs Label)')
    plt.tight_layout()
    fname = name.replace('/', '_')
    os.makedirs('./results/visualization', exist_ok=True)
    plt.savefig(f'./results/visualization/surface_feature_label_{fname}.png')
    plt.close()

# -------------------------------
# 2D plot: positive + negative pair
# -------------------------------
posneg_curves = {}
for loss_cls, name in loss_classes:
    losses = []
    loss_fn = loss_cls()
    for ct in cos_theta:
        anchor = torch.zeros((1, feat_dim))
        anchor[0, 0] = 1.0
        positive = torch.zeros((1, feat_dim))
        positive[0, 0] = ct
        positive[0, 1] = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
        negative = torch.zeros((1, feat_dim))
        negative[0, 0] = -1.0
        anchor = torch.nn.functional.normalize(anchor, dim=1)
        positive = torch.nn.functional.normalize(positive, dim=1)
        negative = torch.nn.functional.normalize(negative, dim=1)
        features = torch.cat([anchor, positive, negative], dim=0)
        labels = torch.tensor([0, 0, 1])
        try:
            loss = loss_fn(features, labels, sigma=0).item()
        except Exception:
            loss = np.nan
        losses.append(loss)
    # 평행이동: θ=0에서의 loss를 0으로 맞춤
    offset = losses[0] if not np.isnan(losses[0]) else 0
    losses_shifted = [l - offset if not np.isnan(l) else np.nan for l in losses]
    posneg_curves[name] = losses_shifted


# 스타일 매핑
style_map = {
    'SupCon_Loss': {'color': 'blue', 'linestyle': '-'},
    'NegDot_Loss': {'color': 'blue', 'linestyle': '--'},
    'MeanNegDot_Loss': {'color': 'blue', 'linestyle': ':'},
    'Weighted_SupConLoss': {'color': 'green', 'linestyle': '-'},
    'NegWeighted_SupConLoss': {'color': 'green', 'linestyle': '--'},
    'ArcCos_Loss': {'color': 'orange', 'linestyle': '-'},
    'NegArcCos_Loss': {'color': 'orange', 'linestyle': '--'},
    'DynamicMargin_Loss': {'color': 'deepskyblue', 'linestyle': '-'},
    'NegDynamicMargin_Loss': {'color': 'deepskyblue', 'linestyle': '--'},
}

plt.figure(figsize=(10, 7))
for name, losses in posneg_curves.items():
    style = style_map.get(name, {'color': 'black', 'linestyle': '-'})
    plt.plot(theta * 180 / np.pi, losses, label=name, color=style['color'], linestyle=style['linestyle'], linewidth=2)
plt.xlabel('Theta (degree)')
plt.ylabel('Loss (pos+neg pair)')
plt.title('Loss vs. Angle (Anchor-Positive, Anchor-Negative)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/visualization/loss_vs_theta_posneg.png')

for name, losses in posneg_curves.items():
    plt.figure(figsize=(8, 5))
    plt.plot(theta * 180 / np.pi, losses, label=name)
    plt.xlabel('Theta (degree)')
    plt.ylabel('Loss (pos+neg pair)')
    plt.title(f'{name} Loss vs. Angle (Pos+Neg Pair)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = name.replace('/', '_')
    plt.savefig(f'./results/visualization/loss_vs_theta_posneg_{fname}.png')
    plt.close()
