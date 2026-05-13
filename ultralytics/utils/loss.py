# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils import LOGGER

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
import numpy as np
import cv2
import matplotlib.pyplot as plt

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class HardNegativeBuffer:
    """Lightweight FIFO buffer of hard-negative sample identifiers."""

    def __init__(self, max_size=10000):
        """Initialize a fixed-size hard-negative buffer."""
        self.max_size = max(int(max_size), 0)
        self.queue = deque()
        self.counts = {}

    def __len__(self):
        """Return the number of buffered entries."""
        return len(self.queue)

    def add(self, samples):
        """Add sample identifiers and evict old entries in FIFO order."""
        if self.max_size <= 0:
            return
        for sample in samples:
            if sample is None:
                continue
            self.queue.append(sample)
            self.counts[sample] = self.counts.get(sample, 0) + 1
            while len(self.queue) > self.max_size:
                oldest = self.queue.popleft()
                count = self.counts.get(oldest, 0) - 1
                if count > 0:
                    self.counts[oldest] = count
                else:
                    self.counts.pop(oldest, None)

    def sample(self, k):
        """Sample up to k unique identifiers from the buffer."""
        if k <= 0 or not self.counts:
            return []
        keys = list(self.counts)
        if k >= len(keys):
            return keys
        return random.sample(keys, k)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
    

class RPFeatureExtractor:
    """
    基于RP特征的对象特征提取器。
    
    RP是输入检测器前的原始特征图，通常是backbone的输出。
    这个提取器基于真实边界框或预测边界框，直接从RP特征图上提取对象特征。
    """
    
    def __init__(self, use_gt_boxes=True, feature_type='roi', roi_size=7):
        """
        初始化RP特征提取器。
        
        Args:
            use_gt_boxes (bool): 是否使用真实边界框
            feature_type (str): 特征提取类型，可选 'center'(中心点) 或 'roi'(区域池化)
            roi_size (int): ROI池化后的大小
        """
        self.use_gt_boxes = use_gt_boxes
        self.feature_type = feature_type
        self.roi_size = roi_size
        
    def extract_center_features(self, rp_features, boxes, img_size, device):
        """
        通过边界框中心点从RP特征图中提取特征。
        
        Args:
            rp_features (torch.Tensor): RP特征图，shape [batch_size, channels, height, width]
            boxes (torch.Tensor): 边界框坐标 (xyxy格式)
            img_size (tuple): 原始图像尺寸 (height, width)
            device (torch.device): 计算设备
            
        Returns:
            torch.Tensor: 中心点特征，shape [num_boxes, channels]
        """
        if boxes.shape[0] == 0:
            return torch.zeros((0, rp_features.shape[1]), device=device)
            
        # 获取特征图尺寸
        _, channels, height, width = rp_features.shape
        
        # 计算中心点坐标
        center_x = ((boxes[:, 0] + boxes[:, 2]) / 2)
        center_y = ((boxes[:, 1] + boxes[:, 3]) / 2)
        
        # 将原图坐标映射到特征图坐标
        #img_size_tensor = torch.tensor(img_size, device=device)
        center_x = (center_x * width / img_size[1]).long().clamp(0, width - 1)
        center_y = (center_y * height / img_size[0]).long().clamp(0, height - 1)
        
        # 提取中心点特征
        batch_idx = boxes[:, -1].long() if boxes.shape[1] > 4 else torch.zeros_like(center_x)
        features = rp_features[batch_idx, :, center_y, center_x]
        
        return features
    
    def extract_roi_features(self, rp_features, boxes, img_size, device):
        """
        通过ROI池化从RP特征图中提取区域特征。
        
        Args:
            rp_features (torch.Tensor): RP特征图，shape [batch_size, channels, height, width]
            boxes (torch.Tensor): 边界框坐标 [batch_idx, bboxes, cls]
            img_size (tuple): 原始图像尺寸 (height, width)
            device (torch.device): 计算设备
            
        Returns:
            torch.Tensor: 池化后的区域特征，shape [num_boxes, channels]
        """
        if boxes.shape[0] == 0:
            return torch.zeros((0, rp_features.shape[1]), device=device)
            
        # 获取特征图尺寸
        batch_size, channels, height, width = rp_features.shape
        
        # 归一化边界框坐标到[0,1]范围
        #img_size_tensor = torch.tensor(img_size, device=device)
        norm_boxes = boxes.clone()
        norm_boxes[:, [1, 3]] /= img_size[1]  # x坐标除以宽度
        norm_boxes[:, [2, 4]] /= img_size[0]  # y坐标除以高度
        
        # 构建ROI批次索引
        batch_idx = boxes[:, -1].long() if boxes.shape[1] > 4 else torch.zeros_like(boxes[:, 0])
        rois = torch.cat([batch_idx.unsqueeze(1), norm_boxes[:, :4]], dim=1)
        rois = torch.cat([batch_idx.unsqueeze(1), norm_boxes[:, 1:5]], dim=1)
        # 执行ROI池化
        try:
            from torchvision.ops import roi_align
            roi_features = roi_align(rp_features, rois, (self.roi_size, self.roi_size))
        except ImportError:
            # 简化的替代方案
            roi_features = []
            for i in range(len(rois)):
                b = batch_idx[i]
                x1, y1, x2, y2 = (norm_boxes[i, :4] * torch.tensor([width, height, width, height],
                                                               device=device)).long()
                # 裁剪并调整大小
                if x1 == x2:
                    x2 = x1 + 1
                if y1 == y2:
                    y2 = y1 + 1
                    
                crop = rp_features[b, :, y1:y2, x1:x2]
                if 0 not in crop.shape[1:]:  # 确保裁剪区域有效
                    pool = F.adaptive_avg_pool2d(crop, (1, 1))
                    roi_features.append(pool)
            
            if roi_features:
                roi_features = torch.cat(roi_features, dim=0)
            else:
                roi_features = torch.zeros((0, channels, 1, 1), device=device)
        
        # 转换为特征向量
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        return roi_features
    
    def extract_features_from_gt(self, rp_features, batch, img_size, device):
        """
        从真实边界框中提取RP特征。
        
        Args:
            rp_features (torch.Tensor): RP特征图
            batch (dict): 批次数据，包含"batch_idx", "cls", "bboxes"等
            img_size (tuple): 原始图像尺寸 (height, width)
            device (torch.device): 计算设备
            
        Returns:
            tuple: (features, labels) - 提取的特征和对应的类别标签
        """
        if "batch_idx" not in batch or "cls" not in batch or "bboxes" not in batch:
            return torch.zeros((0, rp_features.shape[1]), device=device), torch.zeros((0,), device=device, dtype=torch.long)
            
        # 构建GT框信息：[batch_idx, x1, y1, x2, y2, class_id]
        batch_idx = batch["batch_idx"].view(-1, 1)
        cls = batch["cls"].view(-1, 1)
        
        # # 处理边界框格式 - 保证使用xyxy格式
        # bboxes = batch["bboxes"].clone()
        # #if bboxes.shape[1] == 4:  # 如果是xywh格式
        #     # 中心点坐标和宽高转换为左上右下
        # x1y1 = bboxes[:, :2] - bboxes[:, 2:] / 2
        # x2y2 = bboxes[:, :2] + bboxes[:, 2:] / 2
        # bboxes = torch.cat([x1y1, x2y2], dim=1)
        bboxes = xywh2xyxy(batch["bboxes"])
        # 将相对坐标转换为绝对坐标
        #img_size_tensor = torch.tensor(img_size, device=device)
        bboxes[:, [0, 2]] *= img_size[1]  # x坐标乘以宽度
        bboxes[:, [1, 3]] *= img_size[0]  # y坐标乘以高度

#         import matplotlib.patches as patches

# # 假设 batch["img"] 是一个 PyTorch 张量
#         image = batch["img"][0].permute(1, 2, 0).cpu().numpy()  # 转换为 NumPy 格式

#         # 创建图像显示
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)

#         # 绘制边界框
#         for bbox in bboxes:
#             x1, y1, x2, y2 = map(int, bbox[:4])
#             # 创建矩形
#             rect = patches.Rectangle(
#                 (x1, y1), 
#                 x2 - x1,  # width
#                 y2 - y1,  # height
#                 linewidth=2,
#                 edgecolor='g',
#                 facecolor='none'
#             )
#             plt.gca().add_patch(rect)
#             break
#         plt.axis("off")
#         plt.show()
        

        gt_info = torch.cat([batch_idx, bboxes, cls], dim=1).to(device)
        
        # 排除无效样本（边界框为零或负值的）
        valid_mask = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        if not valid_mask.all():
            gt_info = gt_info[valid_mask]
            
        # 根据特征类型提取特征
        if self.feature_type == 'center':
            features = self.extract_center_features(rp_features, gt_info, img_size, device)
        else:  # 'roi'
            features = self.extract_roi_features(rp_features, gt_info, img_size, device)
            
        # 提取类别标签
        labels = gt_info[:, -1].long()
        
        return features, labels



class RPContrastiveLoss(nn.Module):
    """
    基于RP特征的对象级别对比损失。
    
    这个损失直接从backbone输出的RP特征图中提取对象特征，
    而不是使用检测器的后续特征，从而更直接地改进backbone的特征表示能力。
    """
    
    def __init__(
        self,
        temperature=0.5,
        feature_dim=256,
        num_classes=80,
        memory_size=1024,
        use_memory_bank=True,
        use_gt_boxes=True,
        feature_type='roi',
        roi_size=5,
        encode_dim = 1024
    ):
        """
        初始化RP对比损失。
        
        Args:
            temperature (float): 温度参数
            feature_dim (int): 特征维度（应与RP特征的通道数匹配）
            num_classes (int): 类别数量
            memory_size (int): 内存库大小
            use_memory_bank (bool): 是否使用内存特征库
            use_gt_boxes (bool): 是否使用真实边界框
            feature_type (str): 特征提取类型 ('center' 或 'roi')
        """
        super().__init__()
        self.temperature = temperature
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_memory_bank = use_memory_bank
        
        # 特征提取器
        self.feature_extractor = RPFeatureExtractor(
            use_gt_boxes=use_gt_boxes,
            feature_type=feature_type,
            roi_size=5
        )
        
        self.dim = 256
        self.hidden_dim = 2048
        # 初始化内存特征库
        if self.use_memory_bank:
            self.register_buffer('memory_bank', torch.zeros(num_classes, memory_size, self.dim))
            self.register_buffer('memory_count', torch.zeros(num_classes, dtype=torch.long))
            self.memory_size = memory_size
        
        # 类别权重
        self.register_buffer('class_weights', torch.ones(num_classes))
        self.register_buffer('class_freq', torch.zeros(num_classes))
        self.total_samples = 0


        self.feature_projection = nn.Sequential(
            nn.Linear(encode_dim, self.hidden_dim),  # 保持输入输出维度一致，仅用于示例
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.dim)   # 最终输出到预设维度
        )
            
    def update_memory_bank(self, features, labels):
        """更新内存特征库"""
        if not self.use_memory_bank:
            return
            
        for i in range(len(labels)):
            label = labels[i].item()
            if label >= self.num_classes:
                continue
                
            # 更新类别频率
            self.class_freq[label] += 1
            self.total_samples += 1
            
            # 确定在内存库中的位置
            idx = self.memory_count[label] % self.memory_size
            self.memory_bank[label, idx] = features[i].detach().clone()
            self.memory_count[label] += 1
            
        # 更新类别权重
        if self.total_samples > 0:
            cls_prob = self.class_freq / self.total_samples
            cls_prob = torch.clamp(cls_prob, min=1e-8)
            self.class_weights = 1.0 / torch.sqrt(cls_prob + 1e-8)
            self.class_weights = self.class_weights / self.class_weights.mean()
    
    def get_memory_features(self, labels):
        """从内存库获取特征"""
        if not self.use_memory_bank:
            return None, None
            
        unique_labels = torch.unique(labels)
        all_features = []
        all_labels = []
        
        for label in unique_labels:
            if label >= self.num_classes:
                continue
                
            count = min(self.memory_count[label].item(), self.memory_size)
            if count <= 0:
                continue
                
            features = self.memory_bank[label, :count]
            all_features.append(features)
            all_labels.append(torch.full((count,), label, device=labels.device))
        
        if not all_features:
            return None, None
            
        memory_features = torch.cat(all_features, dim=0)
        memory_labels = torch.cat(all_labels, dim=0)
        
        return memory_features, memory_labels
    
    def forward(self, rp_features, batch, img_size):
        """
        前向传播计算对比损失。
        
        Args:
            rp_features (torch.Tensor): RP特征图，shape [batch_size, channels, height, width]
            batch (dict): 批次数据
            img_size (tuple): 原始图像尺寸 (height, width)
            
        Returns:
            torch.Tensor: 对比损失值
        """
        device = rp_features.device
        
        # 检查特征维度是否与初始化时指定的一致
        
        
        # 使用GT框提取特征
        features, labels = self.feature_extractor.extract_features_from_gt(
            rp_features, batch, img_size, device
        )

       
        self.feature_projection=self.feature_projection.to(features.device)
        features = self.feature_projection(features)
        # 如果有效特征数量不足，返回零损失
        if features.shape[0] <= 1:
            return torch.tensor(0.0, device=device)
        
        # 标准化特征
        features = features + 1e-8
        features = F.normalize(features, p=2, dim=1)
        if self.use_memory_bank and features.shape[-1] != self.memory_bank.shape[-1]:
            self.feature_dim = features.shape[1]
            
            # 如果使用内存库，需要重置
            if self.use_memory_bank:
                self.register_buffer('memory_bank', torch.zeros(self.num_classes, self.memory_size, self.feature_dim, 
                                                              device=device))
                self.register_buffer('memory_count', torch.zeros(self.num_classes, dtype=torch.long, device=device))
        # 更新内存特征库并获取额外样本
        self.memory_bank = self.memory_bank.to(device)
        self.memory_count = self.memory_count.to(device)
        if self.use_memory_bank:
            self.update_memory_bank(features, labels)
            memory_features, memory_labels = self.get_memory_features(labels)
            
            if memory_features is not None:
                aug_features = torch.cat([features, memory_features], dim=0)
                aug_labels = torch.cat([labels, memory_labels], dim=0)
            else:
                aug_features = features
                aug_labels = labels
        else:
            aug_features = features
            aug_labels = labels
        
        # 检查每个类别的样本数
        unique_labels, label_counts = torch.unique(aug_labels, return_counts=True)
        valid_classes = unique_labels[label_counts >= 2]  # 每个类别至少需要2个样本
        
        # 如果没有有效类别，返回零损失
        if len(valid_classes) == 0:
            return torch.tensor(0.0, device=device)
        
        # 只保留有效类别的样本
        valid_mask = torch.zeros_like(aug_labels, dtype=torch.bool)
        for cls in valid_classes:
            valid_mask = valid_mask | (aug_labels == cls)
            
        valid_features = aug_features[valid_mask]
        valid_labels = aug_labels[valid_mask]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(valid_features, valid_features.T) / self.temperature
        
        # 创建正负样本掩码
        labels_matrix = valid_labels.view(-1, 1)
        positive_mask = (labels_matrix == valid_labels.view(1, -1)).float()
        
        # 移除自相似度（对角线）
        positive_mask = positive_mask - torch.eye(positive_mask.shape[0], device=device)
        
        # 应用类别权重
        sample_weights = torch.ones_like(valid_labels, dtype=torch.float)
        for i, label in enumerate(valid_labels):
            if label < self.num_classes:
                sample_weights[i] = self.class_weights[label]
                
        # 计算加权对比损失
        exp_logits = torch.exp(similarity_matrix)
        exp_logits = exp_logits - torch.diag(torch.diag(exp_logits))  # 移除对角线
        
        # 计算正样本对的损失
        pos_exp_sum = torch.sum(exp_logits * positive_mask, dim=1)
        all_exp_sum = torch.sum(exp_logits, dim=1)
        
        # 处理可能的零值
        valid_pos = pos_exp_sum > 0
        if not valid_pos.any():
            return torch.tensor(0.0, device=device)
            
        pos_probs = torch.zeros_like(pos_exp_sum)
        pos_probs[valid_pos] = pos_exp_sum[valid_pos] / (all_exp_sum[valid_pos] + 1e-8)
        
        log_probs = torch.zeros_like(pos_probs)
        log_probs[valid_pos] = torch.log(pos_probs[valid_pos] + 1e-8)
        
        loss = -log_probs * sample_weights
        
        # 返回加权平均损失
        return loss.sum() / (sample_weights.sum() + 1e-8)
    

class v8CLloss:
    def __init__(self, model, tal_topk=10,contrastive_weight=0.5,
        memory_size=1024,
        use_gt_boxes=True,
        feature_type='roi',
        roi_size=5,
        inter_dim=256,):  
        # model must be de-paralleled
        
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # self.contrastive_loss = RobustContrastiveLoss(
        #     temperature=0.5,
        #     feature_dim=256,  # 可以根据模型调整
        #     num_classes=self.nc,
        #     memory_size=memory_size,
        #     use_memory_bank=use_memory_bank
        # )

        # try:
        #     if hasattr(model.model, 'backbone'):
        #         # 尝试从主干网络获取
        #         feature_layers = [m for m in model.model.backbone.modules() if isinstance(m, nn.Conv2d)]
        #         feature_dim = feature_layers[-1].out_channels if feature_layers else 256
        #     else:
        #         feature_dim = m.reg_max * 4  # 默认值
        # except:
        #     feature_dim = m.reg_max * 4  # 默认值
        feature_dim = inter_dim  # 直接使用inter_dim作为特征维度

        self.rp_contrastive = RPContrastiveLoss(
            temperature=0.5,
            feature_dim=feature_dim,  # 会在第一次运行时动态调整
            num_classes=self.nc,
            memory_size=memory_size,
            use_memory_bank=True,
            use_gt_boxes=use_gt_boxes,
            feature_type=feature_type,
            encode_dim=roi_size*roi_size*inter_dim,  # 可以根据需要调整
        )
        

        
        self.contrastive_weight = contrastive_weight

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, rp, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() # b*8400*nc
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  
        # 返回list anchorpoints 每个尺度的anchor点坐标和对应的stride 例 [ 0.5000,  0.5000] 到  [39.5000, 39.5000],
        # 和每个尺度对应的stride [8, 16, 32]，每个尺度的stride是一样的 维度是  6400*2, 1600*2, 400*2 和 6400*1,  1600, 400

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]) # 将 max*6 维度的target信息
        # 变成 batch*max*5 维度的target信息
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) #b*max*1 #通过检测最大值是否为0来确认 是不是有效的target

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4) 通过预测一个regmax的distribution的期望来预测相对与anchor points的
        # 左上角和右下的偏移量
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

       
        # if isinstance(rp, list) and len(rp)>=3:
        #     c_features = rp[0]  # [b, 256, 20, 20]
        #     if "cls" in batch and batch["cls"] is not None:
        #         labels = batch["cls"].to(self.device).flatten()
        #         valid_mask = labels >= 0  # 过滤无效标签
        rp = rp[0] if isinstance(rp, list) else rp  # [b, 256, 20, 20]
        if rp is not None and isinstance(rp, torch.Tensor):
            # 获取图像尺寸
            if hasattr(self, 'stride'):
                imgsz = torch.tensor(rp.shape[2:], device=self.device) * self.stride[0]
            else:
                # 如果无法获取stride，使用特征图尺寸的估计
                imgsz = torch.tensor([s * 8 for s in rp.shape[2:]], device=self.device)
            
            # 计算对比损失
            cl_loss = self.rp_contrastive(rp, batch, (imgsz[0].item(), imgsz[1].item()))
            
            # 组合损失
            total_loss = loss.sum() * batch_size + self.contrastive_weight * cl_loss
            
            # 添加对比损失到记录项
            loss_items = torch.cat([loss.detach(), cl_loss.detach().unsqueeze(0)])
            return total_loss, loss_items

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        self.model = model
        m = model.model[-1]  # Detect() module
        if hasattr(m, "configure_moco"):
            m.configure_moco(
                queue_size=getattr(h, "moco_queue_size", None),
                roi_output_size=getattr(h, "moco_roi_output_size", None),
            )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.buffer_size = int(getattr(h, "buffer_size", 10000))
        self.hard_neg_weight = float(getattr(h, "hard_neg_weight", 2.0))
        self.conf_threshold = float(getattr(h, "conf_threshold", 0.5))
        self.topk_ratio = float(getattr(h, "topk_ratio", 0.1))
        self.hard_neg_mode = str(getattr(h, "hard_neg_mode", "class_specific")).lower()
        if self.hard_neg_mode not in {"class_specific", "class_agnostic"}:
            LOGGER.warning(f"Unknown hard_neg_mode={self.hard_neg_mode}, falling back to class_specific.")
            self.hard_neg_mode = "class_specific"
        self.manual_hard_neg_weight = float(getattr(h, "manual_hard_neg_weight", self.hard_neg_weight))
        self.hard_neg_class_id = int(getattr(h, "hard_neg_class_id", min(self.nc - 1, 1)))
        self.manual_hard_neg_class_id = int(getattr(h, "manual_hard_neg_class_id", 0))
        self.notree_neg_weight = float(getattr(h, "notree_neg_weight", 1.5))
        self.notree_conf_threshold = float(getattr(h, "notree_conf_threshold", self.conf_threshold))
        self.notree_topk_ratio = float(getattr(h, "notree_topk_ratio", self.topk_ratio))
        self.hard_neg_iou_threshold = float(getattr(h, "hard_neg_iou_threshold", 0.1))
        self.hard_neg_log_interval = int(getattr(h, "hard_neg_log_interval", 50))
        self.hard_neg_vis_max_store = int(getattr(h, "hard_neg_vis_max_store", 32))
        self.hard_neg_vis_per_batch = int(getattr(h, "hard_neg_vis_per_batch", 2))
        self.hard_neg_vis_samples = int(getattr(h, "hard_neg_vis_samples", 16))
        self.hard_neg_buffer = HardNegativeBuffer(self.buffer_size)
        self.hard_neg_visuals = deque(maxlen=max(self.hard_neg_vis_max_store, 0))
        self.hard_neg_step = 0
        self.total_mined_hard_neg = 0
        self.total_buffered_hard_neg = 0
        self.total_manual_hard_neg = 0
        self.total_notree_neg = 0
        self.last_hard_neg_stats = {
            "step": 0,
            "batch_mined": 0,
            "batch_sicktree_fp": 0,
            "batch_notree_neg": 0,
            "batch_hard_images": 0,
            "batch_buffered": 0,
            "batch_manual": 0,
            "buffer_size": 0,
        }

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _batch_paths(self, batch, batch_size):
        """Return stable per-image identifiers for the current batch."""
        paths = batch.get("im_file")
        if paths is None:
            return [str(i) for i in range(batch_size)]
        return [str(path) for path in paths]

    def _manual_hard_neg_image_mask(self, batch, batch_size):
        """Return a per-image manual hard-negative mask."""
        cls = batch.get("cls")
        batch_idx = batch.get("batch_idx")
        if cls is not None and batch_idx is not None:
            cls = cls.view(-1).to(self.device)
            batch_idx = batch_idx.view(-1).long().to(self.device)
            manual_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            manual_gt_mask = cls == self.manual_hard_neg_class_id
            if manual_gt_mask.any():
                manual_mask[batch_idx[manual_gt_mask]] = True
        else:
            manual_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for key in ("hard_neg_mask", "manual_hard_neg_mask", "is_hard_negative"):
            mask = batch.get(key)
            if mask is None:
                continue
            mask = torch.as_tensor(mask, device=self.device).view(-1).bool()
            if mask.numel() == batch_size:
                manual_mask |= mask
        return manual_mask

    def _positive_target_image_mask(self, batch, batch_size):
        """Return a per-image mask for images containing the target positive class."""
        cls = batch.get("cls")
        batch_idx = batch.get("batch_idx")
        if cls is None or batch_idx is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        cls = cls.view(-1).to(self.device)
        batch_idx = batch_idx.view(-1).long().to(self.device)
        target_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        target_gt_mask = torch.ones_like(cls, dtype=torch.bool) if self.hard_neg_mode == "class_agnostic" else cls == self.hard_neg_class_id
        if target_gt_mask.any():
            target_mask[batch_idx[target_gt_mask]] = True
        return target_mask

    def _select_topk_mask(self, scores, candidate_mask):
        """Select the top-k anchors by detached classification loss."""
        return self._select_topk_mask_with_ratio(scores, candidate_mask, self.topk_ratio)

    def _select_topk_mask_with_ratio(self, scores, candidate_mask, ratio):
        """Select the top-k anchors by detached classification loss using a custom ratio."""
        if ratio <= 0:
            return torch.zeros_like(candidate_mask)

        flat_mask = candidate_mask.view(-1)
        num_candidates = int(flat_mask.sum().item())
        if num_candidates == 0:
            return torch.zeros_like(candidate_mask)

        k = max(int(num_candidates * ratio), 1)
        flat_scores = scores.view(-1)
        selected = torch.zeros_like(flat_mask)
        candidate_idx = flat_mask.nonzero(as_tuple=False).squeeze(1)
        topk_idx = candidate_idx[flat_scores[candidate_idx].topk(min(k, num_candidates)).indices]
        selected[topk_idx] = True
        return selected.view_as(candidate_mask)

    def _buffer_image_mask(self, batch, batch_size):
        """Sample buffered hard-negative images and mark those present in the current batch."""
        if len(self.hard_neg_buffer) == 0:
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        sample_ids = set(self.hard_neg_buffer.sample(batch_size))
        if not sample_ids:
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        return torch.tensor(
            [path in sample_ids for path in self._batch_paths(batch, batch_size)], dtype=torch.bool, device=self.device
        )

    def _max_iou_to_target_gt(self, pred_boxes, gt_boxes, gt_labels, mask_gt):
        """Return per-anchor max IoU to target-class GT boxes for each image."""
        max_iou = pred_boxes.new_zeros(pred_boxes.shape[:2])
        if self.hard_neg_iou_threshold <= 0:
            return max_iou

        for i in range(pred_boxes.shape[0]):
            valid_gt = mask_gt[i, :, 0].bool()
            if self.hard_neg_mode == "class_specific":
                target_label = gt_labels.new_tensor(float(self.hard_neg_class_id))
                valid_gt &= gt_labels[i, :, 0] == target_label
            if not valid_gt.any():
                continue
            target_gt_boxes = gt_boxes[i, valid_gt]
            pred_xyxy = pred_boxes[i].unsqueeze(1)  # [N, 1, 4]
            gt_xyxy = target_gt_boxes.unsqueeze(0)  # [1, M, 4]
            inter_w = (pred_xyxy[..., 2].minimum(gt_xyxy[..., 2]) - pred_xyxy[..., 0].maximum(gt_xyxy[..., 0])).clamp_(0)
            inter_h = (pred_xyxy[..., 3].minimum(gt_xyxy[..., 3]) - pred_xyxy[..., 1].maximum(gt_xyxy[..., 1])).clamp_(0)
            inter = inter_w * inter_h
            pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp_(0) * (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp_(0)
            gt_area = (gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp_(0) * (gt_xyxy[..., 3] - gt_xyxy[..., 1]).clamp_(0)
            union = pred_area + gt_area - inter + 1e-7
            max_iou[i] = (inter / union).max(dim=1).values
        return max_iou

    def _collect_hard_negative_visuals(
        self, batch, mined_mask, pred_conf, pred_bboxes, stride_tensor, sample_tags
    ):
        """Store a few hard-negative full-image views for end-of-training visualization."""
        if self.hard_neg_vis_max_store <= 0 or self.hard_neg_vis_per_batch <= 0:
            return
        if not mined_mask.any():
            return

        pred_boxes_img = (pred_bboxes.detach() * stride_tensor).round().long()
        img_h, img_w = batch["img"].shape[2:]
        candidate_scores = pred_conf.masked_fill(~mined_mask, -1)
        flat_scores = candidate_scores.view(-1)
        valid = flat_scores > -1
        if not valid.any():
            return

        topk = min(int(valid.sum().item()), self.hard_neg_vis_per_batch)
        top_idx = flat_scores.topk(topk).indices.tolist()
        images = (batch["img"].detach().clamp(0, 1) * 255).to(torch.uint8).cpu()
        paths = self._batch_paths(batch, batch["img"].shape[0])

        for flat_idx in top_idx:
            b_idx = flat_idx // mined_mask.shape[1]
            a_idx = flat_idx % mined_mask.shape[1]
            x1, y1, x2, y2 = pred_boxes_img[b_idx, a_idx].tolist()
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 2, min(x2, img_w))
            y2 = max(y1 + 2, min(y2, img_h))
            full_image = images[b_idx].permute(1, 2, 0).numpy().copy()
            full_image = cv2.resize(full_image, (160, 160), interpolation=cv2.INTER_LINEAR)
            scale_x, scale_y = 160.0 / img_w, 160.0 / img_h
            box = (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            )
            self.hard_neg_visuals.append(
                {
                    "image": full_image,
                    "box": box,
                    "score": float(pred_conf[b_idx, a_idx].item()),
                    "path": Path(paths[b_idx]).name,
                    "tag": sample_tags[b_idx][a_idx],
                }
            )

    def _update_hard_negative_stats(
        self, hard_neg_mask, notree_neg_mask, buffered_topk_mask, manual_hard_neg_image_mask
    ):
        """Update running hard-negative statistics and optionally log them."""
        self.hard_neg_step += 1
        batch_sicktree_fp = int(hard_neg_mask.sum().item())
        batch_notree_neg = int(notree_neg_mask.sum().item())
        batch_mined = batch_sicktree_fp + batch_notree_neg
        batch_hard_images = int((hard_neg_mask | notree_neg_mask).any(1).sum().item())
        batch_buffered = int(buffered_topk_mask.sum().item())
        batch_manual = int((manual_hard_neg_image_mask.sum().item()) if manual_hard_neg_image_mask is not None else 0)

        self.total_mined_hard_neg += batch_mined
        self.total_buffered_hard_neg += batch_buffered
        self.total_manual_hard_neg += batch_manual
        self.total_notree_neg += batch_notree_neg
        self.last_hard_neg_stats = {
            "step": self.hard_neg_step,
            "batch_mined": batch_mined,
            "batch_sicktree_fp": batch_sicktree_fp,
            "batch_notree_neg": batch_notree_neg,
            "batch_hard_images": batch_hard_images,
            "batch_buffered": batch_buffered,
            "batch_manual": batch_manual,
            "buffer_size": len(self.hard_neg_buffer),
        }

        if self.hard_neg_log_interval > 0 and self.hard_neg_step % self.hard_neg_log_interval == 0:
            LOGGER.info(
                "Hard negatives: "
                f"step={self.hard_neg_step}, "
                f"mined={batch_mined}, "
                f"sicktree_fp={batch_sicktree_fp}, "
                f"notree_neg={batch_notree_neg}, "
                f"hard_images={batch_hard_images}, "
                f"buffer_hit={batch_buffered}, "
                f"manual={batch_manual}, "
                f"buffer_size={len(self.hard_neg_buffer)}"
            )

    def get_hard_negative_stats(self):
        """Return cumulative and last-step hard-negative statistics."""
        return {
            **self.last_hard_neg_stats,
            "total_mined": self.total_mined_hard_neg,
            "total_buffered": self.total_buffered_hard_neg,
            "total_manual": self.total_manual_hard_neg,
            "total_notree_neg": self.total_notree_neg,
            "visual_samples": len(self.hard_neg_visuals),
        }

    def save_hard_negative_visuals(self, save_dir, max_samples=None):
        """Save a simple grid of mined hard-negative crops."""
        if not self.hard_neg_visuals:
            return None

        samples = list(self.hard_neg_visuals)[-min(max_samples or self.hard_neg_vis_samples, len(self.hard_neg_visuals)) :]
        tile_size, header_h, cols = 160, 22, min(4, max(len(samples), 1))
        rows = math.ceil(len(samples) / cols)
        canvas = np.full((rows * (tile_size + header_h), cols * tile_size, 3), 255, dtype=np.uint8)

        for i, sample in enumerate(samples):
            row, col = divmod(i, cols)
            y0, x0 = row * (tile_size + header_h), col * tile_size
            tile = sample["image"].copy()
            x1, y1, x2, y2 = sample["box"]
            color = (255, 196, 0) if sample["tag"] == "sicktree_fp" else (0, 200, 0) if sample["tag"] == "notree" else (0, 128, 255)
            cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                tile,
                f"s:{sample['score']:.2f}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            canvas[y0 + header_h : y0 + header_h + tile_size, x0 : x0 + tile_size] = tile
            cv2.putText(
                canvas,
                f"{sample['tag']} {sample['path'][:14]}",
                (x0 + 4, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (32, 32, 32),
                1,
                cv2.LINE_AA,
            )

        out_file = Path(save_dir) / "hard_negative_samples.jpg"
        cv2.imwrite(str(out_file), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return out_file

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() # b*8400*nc
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  
        # 返回list anchorpoints 每个尺度的anchor点坐标和对应的stride 例 [ 0.5000,  0.5000] 到  [39.5000, 39.5000],
        # 和每个尺度对应的stride [8, 16, 32]，每个尺度的stride是一样的 维度是  6400*2, 1600*2, 400*2 和 6400*1,  1600, 400

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]) # 将 max*6 维度的target信息
        # 变成 batch*max*5 维度的target信息
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) #b*max*1 #通过检测最大值是否为0来确认 是不是有效的target

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4) 通过预测一个regmax的distribution的期望来预测相对与anchor points的
        # 左上角和右下的偏移量
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))
        bg_mask = ~fg_mask
        pred_scores_sigmoid = pred_scores.detach().sigmoid()
        if self.hard_neg_mode == "class_agnostic":
            hard_neg_cls_loss, hard_neg_class_idx = cls_loss.detach().max(dim=-1)
            hard_neg_conf, _ = pred_scores_sigmoid.max(dim=-1)
            notree_cls_loss = cls_loss[..., self.manual_hard_neg_class_id].detach()
            notree_conf = pred_scores_sigmoid[..., self.manual_hard_neg_class_id]
        else:
            hard_neg_cls_loss = cls_loss[..., self.hard_neg_class_id].detach()
            hard_neg_conf = pred_scores_sigmoid[..., self.hard_neg_class_id]
            hard_neg_class_idx = None
            notree_cls_loss = cls_loss[..., self.manual_hard_neg_class_id].detach()
            notree_conf = pred_scores_sigmoid[..., self.manual_hard_neg_class_id]
        conf_hard_neg_mask = bg_mask & hard_neg_conf.gt(self.conf_threshold)
        mined_topk_mask = self._select_topk_mask(hard_neg_cls_loss, bg_mask)
        hard_neg_mask = conf_hard_neg_mask | mined_topk_mask
        notree_conf_mask = torch.zeros_like(bg_mask) if self.hard_neg_mode == "class_agnostic" else bg_mask & notree_conf.gt(self.notree_conf_threshold)
        notree_ratio = 0.0 if self.hard_neg_mode == "class_agnostic" else self.notree_topk_ratio
        notree_topk_mask = self._select_topk_mask_with_ratio(notree_cls_loss, bg_mask, notree_ratio)
        notree_neg_mask = (notree_conf_mask | notree_topk_mask) & ~hard_neg_mask
        pred_bboxes_img = pred_bboxes.detach() * stride_tensor
        max_iou_to_sicktree_gt = self._max_iou_to_target_gt(pred_bboxes_img, gt_bboxes, gt_labels, mask_gt)
        low_iou_mask = max_iou_to_sicktree_gt.lt(self.hard_neg_iou_threshold)
        hard_neg_mask &= low_iou_mask
        notree_neg_mask &= low_iou_mask

        manual_hard_neg_image_mask = self._manual_hard_neg_image_mask(batch, batch_size)
        positive_target_image_mask = self._positive_target_image_mask(batch, batch_size)
        mined_mask = hard_neg_mask | notree_neg_mask
        pure_hard_neg_image_mask = mined_mask.any(1) & ~positive_target_image_mask

        if self.model.training and self.buffer_size > 0:
            buffered_image_mask = self._buffer_image_mask(batch, batch_size)
            buffered_topk_mask = self._select_topk_mask(
                hard_neg_cls_loss, bg_mask & buffered_image_mask.unsqueeze(1) & low_iou_mask
            )
        else:
            buffered_topk_mask = torch.zeros_like(bg_mask)

        cls_weight = torch.ones_like(cls_loss)
        if self.hard_neg_weight > 1:
            hard_neg_weight = cls_weight.new_full((), self.hard_neg_weight)
            if self.hard_neg_mode == "class_agnostic":
                hard_neg_cls_mask = torch.zeros_like(cls_loss, dtype=torch.bool)
                hard_neg_cls_mask.scatter_(2, hard_neg_class_idx.unsqueeze(-1), True)
                hard_neg_cls_mask &= (hard_neg_mask | buffered_topk_mask).unsqueeze(-1)
                cls_weight = torch.where(hard_neg_cls_mask, hard_neg_weight, cls_weight)
            else:
                cls_weight[..., self.hard_neg_class_id] = torch.where(
                    hard_neg_mask, hard_neg_weight, cls_weight[..., self.hard_neg_class_id]
                )
                cls_weight[..., self.hard_neg_class_id] = torch.where(
                    buffered_topk_mask, hard_neg_weight, cls_weight[..., self.hard_neg_class_id]
                )

        if self.notree_neg_weight > 1:
            notree_weight = cls_weight.new_full((), self.notree_neg_weight)
            cls_weight[..., self.manual_hard_neg_class_id] = torch.where(
                notree_neg_mask, notree_weight, cls_weight[..., self.manual_hard_neg_class_id]
            )

        if self.manual_hard_neg_weight > 1 and manual_hard_neg_image_mask.any():
            manual_weight = cls_weight.new_full((), self.manual_hard_neg_weight)
            manual_mask = bg_mask & manual_hard_neg_image_mask.unsqueeze(1)
            cls_weight[..., self.hard_neg_class_id] = torch.where(
                manual_mask, manual_weight, cls_weight[..., self.hard_neg_class_id]
            )
            cls_weight[..., self.manual_hard_neg_class_id] = torch.where(
                manual_mask, manual_weight, cls_weight[..., self.manual_hard_neg_class_id]
            )

        loss[1] = (cls_loss * cls_weight).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        if self.model.training and self.buffer_size > 0:
            batch_paths = self._batch_paths(batch, batch_size)
            self.hard_neg_buffer.add(
                path for path, is_hard in zip(batch_paths, pure_hard_neg_image_mask.tolist()) if is_hard
            )

        if self.model.training:
            sample_tags = np.full(mined_mask.shape, "bg", dtype=object)
            sample_tags[notree_neg_mask.cpu().numpy()] = "notree"
            sample_tags[hard_neg_mask.cpu().numpy()] = "sicktree_fp"
            self._collect_hard_negative_visuals(
                batch, mined_mask, hard_neg_conf, pred_bboxes, stride_tensor, sample_tags
            )
            self._update_hard_negative_stats(
                hard_neg_mask, notree_neg_mask, buffered_topk_mask, manual_hard_neg_image_mask
            )

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8MoCoDetectionLoss(v8DetectionLoss):
    """结合对象级MoCo对比学习的YOLO检测损失"""

    def __init__(self, model, tal_topk=10, contrastive_weight=0.5):
        """
        初始化 v8MoCoDetectionLoss。
        Args:
            model: YOLO 模型实例。
            tal_topk: 用于目标分配的 top-k 值。
            contrastive_weight: 对比损失的权重。
        """
        super().__init__(model, tal_topk=tal_topk)
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss_fn = nn.CrossEntropyLoss()  # 对比损失函数
        self.moco_start_epoch = int(getattr(self.hyp, "moco_start_epoch", 20))
        self.moco_ramp_epochs = max(int(getattr(self.hyp, "moco_ramp_epochs", 10)), 0)

    def __call__(self, preds, batch, epoch=None):
        """
        计算总损失，包括检测损失和对比损失。
        Args:
            preds: 模型的预测输出。
            batch: 包含目标信息的批次数据。
        Returns:
            total_loss: 总损失。
            loss_items: 各项损失的详细信息。
        """
        # 解包预测值
        if isinstance(preds, tuple) and len(preds) in {6, 7, 9}:
            enqueue_features = None
            enqueue_labels = None
            if len(preds) == 9:
                (det_head_outputs, raw_features, query_features, key_features, object_labels,
                 queue_snapshot, queue_counts, enqueue_features, enqueue_labels) = preds
            elif len(preds) == 7:
                det_head_outputs, raw_features, query_features, key_features, object_labels, queue_snapshot, queue_counts = preds
            else:
                det_head_outputs, raw_features, query_features, key_features, object_labels, queue_snapshot = preds
                queue_counts = None

            # 1. 计算标准检测损失
            det_loss, det_loss_items = super().__call__((raw_features, det_head_outputs), batch)

            # 2. 计算对比损失
            cl_loss = torch.tensor(0.0, device=self.device)  # 初始化对比损失
            contrastive_scale = 1.0
            if epoch is not None:
                if epoch < self.moco_start_epoch:
                    contrastive_scale = 0.0
                elif self.moco_ramp_epochs > 0:
                    contrastive_scale = min((epoch - self.moco_start_epoch + 1) / self.moco_ramp_epochs, 1.0)

            if contrastive_scale > 0 and query_features is not None and key_features is not None and object_labels is not None:
                if query_features.shape[0] > 1:  # 至少需要两个对象特征
                    # 计算对比损失
                    cl_loss = self._compute_contrastive_loss(query_features, key_features, object_labels, queue_snapshot, queue_counts)
                    #cl_loss = cl_loss * contrastive_scale

            # 3. 合并损失
            total_loss = det_loss + self.contrastive_weight * cl_loss* contrastive_scale

            # 4. 返回损失和详细信息
            head = self.model.model[-1]
            enqueue_features = key_features if enqueue_features is None else enqueue_features
            enqueue_labels = object_labels if enqueue_labels is None else enqueue_labels
            if enqueue_features is not None and enqueue_labels is not None and enqueue_features.numel() > 0:
                with torch.no_grad():
                    head._dequeue_and_enqueue(enqueue_features.detach(), enqueue_labels.detach())

            loss_items = torch.cat([det_loss_items, cl_loss.detach().unsqueeze(0)])
            return total_loss, loss_items
        else:
            return super().__call__(preds, batch)

    def _compute_contrastive_loss(self, query_features, key_features, object_labels, queue_snapshot, queue_counts=None):
        """
        计算对象级别的对比损失。
        Args:
            query_features: 查询特征 [N, feature_dim]。
            key_features: 键特征 [N, feature_dim]。
            object_labels: 对象标签 [N]。
            queue_snapshot: MoCo 队列的快照 [num_classes, queue_size, feature_dim]。
        Returns:
            cl_loss: 对比损失值。
        """
        """
        对同类别的 key_features（同 i）和 queue 中同 class 条目都视为 positives，
        其它视为 negatives，计算 Supervised Contrastive Loss。
        """
        device = query_features.device

        # 1. 归一化 query 和 key
        q = F.normalize(query_features.to(device), dim=1)   # [N, D]
        k = F.normalize(key_features.to(device),   dim=1) 
        k = k.detach()  # [N, D]
        N, D = q.shape

        # 2. 准备 memory bank 作为全部负样本
        C, Qsize, _ = queue_snapshot.shape
        neg_feats = queue_snapshot.to(device).view(C * Qsize, D)  # [M, D]

        # 3. 计算相似度并 exponentiate
        temp = getattr(self.hyp, 'temperature', 0.07)
        sim_kk = torch.matmul(q, k.t()) / temp          # [N, N]
        sim_qn = torch.matmul(q, neg_feats.t()) / temp  # [N, M]
        exp_kk = torch.exp(sim_kk)                      # [N, N]
        exp_qn = torch.exp(sim_qn)                      # [N, M]

        # 4. 构造同 batch 正样本掩码（同类别且非自身）
        labels = object_labels.to(device).view(-1, 1)   # [N,1]
        pos_mask = labels.eq(labels.t()).float()   
        batch_neg_mask = 1 - pos_mask 
        # raw_mask = labels.eq(labels.t())                # [N, N]
        # 统计每行同类样本数
        # pos_counts = raw_mask.sum(1)                    # [N]
        # pos_mask = raw_mask.clone()
        # idx = torch.arange(N, device=device)
        # 只有当某行除了自身外还有其它 positives 时，才将对角线置为 False
        # remove_idx = idx[pos_counts > 1]
        # pos_mask[remove_idx, remove_idx] = False

        # 5. 分别求 positives 和 negatives 的总和
        #pos_sum = (exp_kk * pos_mask.float()).sum(1)    # [N]
        #pos_sum = exp_kk.sum(1)  # [N]  # 直接使用所有 positives 的总和

        pos_counts = pos_mask.sum(1)    
        idx = torch.arange(N, device=device)
        # 只有当某行除了自身外还有其它 positives 时，才将对角线置为 False
        remove_idx = idx[pos_counts > 1]
        pos_mask[remove_idx, remove_idx] = 0


        pos_sum = (exp_kk * pos_mask).sum(1)  

        
        batch_neg_sum = (exp_kk * batch_neg_mask).sum(1)  # [N, N]
        queue_labels = torch.arange(C, device=device).unsqueeze(1).repeat(1, Qsize).view(-1)  # [M]
        # neg_mask[i,j]=True 当 queue_labels[j]!=object_labels[i]
        neg_mask = queue_labels.view(1, -1).ne(labels)                   # [N, M]
        queue_neg_sum = (exp_qn * neg_mask.float()).sum(1)  
        neg_sum = queue_neg_sum + batch_neg_sum  # [N]  # 将 batch 内 negatives 和 queue 中 negatives 相加

        # 6. 计算 InfoNCE 损失：−log(pos / (pos + neg))
        eps = 1e-8
        alpha = 1.5
        beta = 0.999

        loss = -torch.log((pos_sum + eps) / (pos_sum + neg_sum + eps))
        unique_labels, counts = torch.unique(object_labels, return_counts=True)
        weights_tensor = 1.0 / (counts.float() + 1e-8)
        weights_tensor =  1.0 / torch.log(alpha + counts.float())
        # effective_num = 1.0 - torch.pow(beta, counts.float())
        # weights = (1.0 - beta) / effective_num
        # weights_tensor =  weights / weights.sum() * len(unique_labels)
        sample_weight = torch.zeros_like(object_labels, dtype=torch.float, device=device)
        for label, w in zip(unique_labels, weights_tensor):
            sample_weight[object_labels == label] = w

        # 然后计算 loss 时乘上权重，再做归一化
        loss = (loss * sample_weight).sum() / (sample_weight.sum()+ 1e-8)
        return loss
                                                     
        #return loss.mean()



        # 3. 构造 pool = all batch-keys (positives) + flat_queue
        pool_feat = torch.cat([k, flat_queue], dim=0)                     # [N+M, D]
        pool_labels = torch.cat([object_labels, queue_labels], dim=0)     # [N+M]

        # 4. 相似度矩阵
        temp = getattr(self.hyp, 'temperature', 0.07)
        sim = torch.matmul(q, pool_feat.t()) / temp                       # [N, N+M]

        # 5. 同/异 类掩码
        labels_q = object_labels.view(-1, 1)                              # [N,1]
        mask_pos = pool_labels.view(1, -1).eq(labels_q)                   # [N, N+M]
        # 排除 query 自身与 key_features 对角线位置
        idx = torch.arange(q.size(0), device=q.device)
        mask_pos[idx, idx] = False

        # 6. 计算 SupCon Loss
        exp_sim = torch.exp(sim)                                          # [N, N+M]
        # 分子：每行同 class 的 exp_sim 求和；分母：整行 exp_sim 求和
        pos_exp = (exp_sim * mask_pos.float()).sum(1)                     # [N]
        all_exp = exp_sim.sum(1)                                          # [N]
        # 避免除 0
        eps = 1e-8
        loss_per_sample = -torch.log((pos_exp + eps) / (all_exp + eps))   # [N]
        return loss_per_sample.mean()
    
    def _stable_moco_contrastive_loss(self, query_features, key_features, object_labels, queue_snapshot, queue_counts=None):
        """Compute a stable supervised contrastive loss using batch and queue positives."""
        device = query_features.device
        q = F.normalize(query_features.to(device), dim=1)
        k = F.normalize(key_features.to(device), dim=1).detach()
        labels = object_labels.to(device).view(-1)
        n, d = q.shape

        if n <= 1:
            return torch.tensor(0.0, device=device)

        c, qsize, _ = queue_snapshot.shape
        queue_feats = F.normalize(queue_snapshot.to(device).view(c * qsize, d), dim=1)
        queue_labels = torch.arange(c, device=device).unsqueeze(1).repeat(1, qsize).view(-1)
        if queue_counts is None:
            queue_valid = torch.ones(c, qsize, dtype=torch.bool, device=device)
        else:
            queue_counts = queue_counts.to(device).clamp_(min=0, max=qsize)
            queue_valid = torch.arange(qsize, device=device).view(1, -1).lt(queue_counts.view(-1, 1))
        queue_valid = queue_valid.view(-1)

        temp = max(float(getattr(self.hyp, "temperature", 0.1)), 1e-6)
        logits_batch = torch.matmul(q, k.t()) / temp
        logits_queue = torch.matmul(q, queue_feats.t()) / temp

        pos_mask_batch = labels.view(-1, 1).eq(labels.view(1, -1))
        pos_mask_batch.fill_diagonal_(False)
        pos_mask_queue = labels.view(-1, 1).eq(queue_labels.view(1, -1))
        pos_mask_queue &= queue_valid.view(1, -1)
        valid_mask = pos_mask_batch.any(dim=1) | pos_mask_queue.any(dim=1)

        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        logits = torch.cat([logits_batch, logits_queue], dim=1)
        pos_mask = torch.cat([pos_mask_batch, pos_mask_queue], dim=1).float()
        valid_logit_mask = torch.cat(
            [torch.ones(n, n, dtype=torch.bool, device=device), queue_valid.view(1, -1).expand(n, -1)],
            dim=1,
        )
        logits = logits.masked_fill(~valid_logit_mask, float("-inf"))
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)

        pos_exp = (exp_logits * pos_mask).sum(dim=1)
        all_exp = exp_logits.sum(dim=1)
        eps = 1e-8
        loss = -torch.log((pos_exp + eps) / (all_exp + eps))
        loss = loss[valid_mask]
        valid_labels = labels[valid_mask]

        unique_labels, counts = torch.unique(valid_labels, return_counts=True)
        sample_weight = torch.ones_like(valid_labels, dtype=torch.float, device=device)
        for label, count in zip(unique_labels, counts.float()):
            sample_weight[valid_labels == label] = 1.0 / (count + eps)

        return (loss * sample_weight).sum() / (sample_weight.sum() + eps)


v8MoCoDetectionLoss._compute_contrastive_loss = v8MoCoDetectionLoss._stable_moco_contrastive_loss

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
