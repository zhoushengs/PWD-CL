# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import copy
import math
from typing import List
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "DetectWithObjectMoCo","DetectWithMoCoBK"


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)
        x_ = [t.clone() for t in x]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return (x_,x)
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

# ... existing imports ...
from torchvision.ops import roi_align
import torch.nn.functional as F

class EncoderWithProjector(nn.Module):
    def __init__(self, in_c, feat_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
            # nn.Linear(hidden_dim, feat_dim, bias=True),
        )

    def forward(self, x):
        feat = self.encoder(x)               # 未 normalize
        z = self.projector(feat)             # 未 normalize
        #z = F.normalize(z, dim=1)            # contrastive 需要
        return z
    
class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn = self.mk(x)                          # [B, S]
        attn = self.softmax(attn)                  # normalize
        out = self.mv(attn)                        # [B, D]
        return out
class EncoderWithProjectorEA(nn.Module):
    def __init__(self, in_c, feat_dim, hidden_dim=64):
        super().__init__()

        # Encoder 和你原来一样
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),

            nn.Conv2d(in_c, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # External Attention + 线性层作为 projector
        self.projector = nn.Sequential(
            ExternalAttention(feat_dim, S=hidden_dim),
            #nn.Linear(feat_dim, feat_dim, bias=False)
        )

    def forward(self, x):
        feat = self.encoder(x)
        z = self.projector(feat)
        z = F.normalize(z, dim=1)    # contrastive 推荐加
        return z



class DetectWithObjectMoCo(Detect):
    def __init__(self, nc=80, ch=(), queue_size=8192*2, momentum=0.999, feature_dim=32, roi_output_size=7):
        """
        Detection head with MoCo for object-level contrastive learning.
        Args:
            nc (int): Number of classes for detection and MoCo queue.
            ch (tuple): Tuple of input channels for each FPN level from the neck.
                        E.g., (c_p3, c_p4, c_p5).
            queue_size (int): Size of the MoCo queue per class.
            momentum (float): Momentum for updating the key encoder.
            feature_dim (int): Dimension of the encoded object features for MoCo.
            roi_output_size (int): Output spatial size of RoIAlign (e.g., 7 for 7x7).
        """
        super().__init__(nc=nc, ch=ch)  # Initialize parent Detect class

        self.feature_dim = feature_dim #ch[0]*2
        self.roi_output_size = roi_output_size
        self.momentum = momentum
        self.queue_size = queue_size
        self.nc = nc
        self.hidden_dim = 2*feature_dim  # Hidden dimension for the transformer decoder
        # Determine the number of channels of the feature map used for RoIAlign.
        # We'll use the feature map from the first FPN level (e.g., P3),
        # which corresponds to ch[0] if ch is (c_p3, c_p4, c_p5).
        if not isinstance(ch, (list, tuple)) or not ch:
            raise ValueError("`ch` must be a non-empty list or tuple of channel numbers.")
        c_feat_map_for_roi = ch[0]
        feature_dim_2 = ch[1]
        # Define Query and Key Encoders
        # Input: [N, c_feat_map_for_roi, roi_output_size, roi_output_size]
        # Output: [N, feature_dim]
        self.dim_map = nn.Sequential(
            nn.Conv2d(feature_dim_2, c_feat_map_for_roi, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_feat_map_for_roi),
            nn.ReLU()
        )
        #common_encoder_layers = EncoderWithProjector(c_feat_map_for_roi, self.feature_dim, self.hidden_dim) 
        # common_encoder_layers = lambda in_c: nn.Sequential(
        #     # nn.Conv2d(in_c, self.feature_dim, kernel_size=3, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.feature_dim),
        #     # nn.ReLU(inplace=True),
        #     # Depthwise Convolution
        #     nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False),
        #     nn.BatchNorm2d(in_c),
        #     nn.ReLU(inplace=True),
        #     # Pointwise Convolution (1x1)
        #     nn.Conv2d(in_c, self.feature_dim, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(self.feature_dim),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Output: [N, feature_dim, 1, 1]
        #     nn.Flatten(),  # Output: [N, feature_dim]
        #     # MoCo projection head
        #     nn.Linear(self.feature_dim, self.hidden_dim, bias=False),
        #     nn.BatchNorm1d(self.hidden_dim), # BN after linear for projection
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(self.hidden_dim, self.feature_dim, bias=True), # Final projection
        #     nn.functional.normalize(dim=1)
        # # )
        # self.query_encoder = common_encoder_layers(c_feat_map_for_roi)
        # self.key_encoder = common_encoder_layers(c_feat_map_for_roi)
        self.query_encoder = EncoderWithProjectorEA(c_feat_map_for_roi, self.feature_dim, self.hidden_dim)
        self.key_encoder = EncoderWithProjectorEA(c_feat_map_for_roi, self.feature_dim, self.hidden_dim)

        # Initialize key_encoder with query_encoTder weights and stop its gradients
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo Queue for negative keys (one queue per class)
        self.register_buffer('queue', torch.randn(self.nc, self.queue_size, self.feature_dim))
        self.queue = F.normalize(self.queue, p=2, dim=2)  # Normalize features in the queue
        self.register_buffer('queue_ptr', torch.zeros(self.nc, dtype=torch.long))

        # Optional: For class frequency statistics (if needed by other parts of your code)
        self.register_buffer('class_freq', torch.zeros(self.nc))
        self.total_samples = 0
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, encoded_key_features, object_labels):
        """
        Dequeue oldest features and enqueue new key features for each class.
        Args:
            encoded_key_features (torch.Tensor): Encoded and L2-normalized key features [num_objects, feature_dim].
            object_labels (torch.Tensor): Object class labels [num_objects].
        """
        for i, label_val in enumerate(object_labels):
            label = int(label_val.item())
            if not (0 <= label < self.nc): # Ensure label is valid
                continue
            
            # Optional: Update class frequency (if you use it elsewhere)
            # self.class_freq[label] += 1
            # self.total_samples += 1
            
            ptr = int(self.queue_ptr[label])
            self.queue[label, ptr] = encoded_key_features[i] # Enqueue the new key feature
            self.queue_ptr[label] = (ptr + 1) % self.queue_size # Move pointer

    def _extract_roi_encoded_features(self,
                                      feature_maps_from_neck: List[torch.Tensor],
                                      bboxes_abs_img_scale: torch.Tensor,
                                      batch_indices_for_roi: torch.Tensor,
                                      current_batch_img_shapes: torch.Tensor,
                                      encoder: nn.Module):
        """
        Extracts RoI features, encodes them, and normalizes.
        Args:
            feature_maps_from_neck (List[torch.Tensor]): List of FPN feature maps [P3, P4, P5,...].
            bboxes_abs_img_scale (torch.Tensor): Absolute GT bounding boxes [N_roi, 4] (xyxy) on original image scale.
            batch_indices_for_roi (torch.Tensor): Batch index for each RoI [N_roi].
            current_batch_img_shapes (torch.Tensor): Shapes of images in the current batch [BatchSize, 2] (H, W).
            encoder (nn.Module): The encoder module (query_encoder or key_encoder).
        Returns:
            torch.Tensor: L2-normalized encoded object features [N_roi, feature_dim].
        """
        if bboxes_abs_img_scale.numel() == 0:
            return torch.empty(0, self.feature_dim, device=feature_maps_from_neck[0].device)

        # Use the P3 feature map (smallest stride, highest resolution) for RoIAlign
        selected_feature_map = feature_maps_from_neck
        _bs, _c_fm, h_fm, w_fm = selected_feature_map.shape

        # selected_feature_map2 = feature_maps_from_neck[1]
        # _bs2, _c_fm2, h_fm2, w_fm2 = selected_feature_map2.shape

        # Ensure all tensors are on the same device
        device = selected_feature_map.device
        bboxes_abs_img_scale    = bboxes_abs_img_scale.to(device)
        batch_indices_for_roi   = batch_indices_for_roi.to(device)
        current_batch_img_shapes = current_batch_img_shapes.to(device)

        H_img, W_img = current_batch_img_shapes[0].item(), current_batch_img_shapes[1].item()
        # bboxes_abs_img_scale * [W,H,W,H] �?像素xywh
        scale_img = torch.tensor([W_img, H_img, W_img, H_img],
                                 device=device, dtype=bboxes_abs_img_scale.dtype)
        pixel_xywh = bboxes_abs_img_scale * scale_img  # [N,4]
        pixel_xyxy = xywh2xyxy(pixel_xywh) 

        norm_xyxy = pixel_xyxy / scale_img        # [N,4]
        fm_scale = torch.tensor([w_fm, h_fm, w_fm, h_fm], device=device)
        rois_fm = norm_xyxy * fm_scale  

        rois_fm[:, 0::2].clamp_(0, w_fm - 1)
        rois_fm[:, 1::2].clamp_(0, h_fm - 1)

        bad_w = rois_fm[:, 2] <= rois_fm[:, 0]
        rois_fm[bad_w, 2] = rois_fm[bad_w, 0] + 1
        bad_h = rois_fm[:, 3] <= rois_fm[:, 1]
        rois_fm[bad_h, 3] = rois_fm[bad_h, 1] + 1

        # rois_fm2 = norm_xyxy * torch.tensor([w_fm2, h_fm2, w_fm2, h_fm2], device=device)
        # rois_fm2[:, 0::2].clamp_(0, w_fm2 - 1)
        # rois_fm2[:, 1::2].clamp_(0, h_fm2 - 1)

        # bad_w2 = rois_fm2[:, 2] <= rois_fm2[:, 0]
        # rois_fm2[bad_w2, 2] = rois_fm2[bad_w2, 0] + 1
        # bad_h2 = rois_fm2[:, 3] <= rois_fm2[:, 1]
        # rois_fm2[bad_h2, 3] = rois_fm2[bad_h2, 1] + 1

        # Scale RoIs from original image coordinates to the selected feature map's coordinates
        
        roi_inputs = torch.cat([batch_indices_for_roi.to(device).unsqueeze(1).float(),
                                rois_fm], dim=1)  # [N,5]

        # roi_inputs2 = torch.cat([batch_indices_for_roi.to(device).unsqueeze(1).float(),
        #                           rois_fm2], dim=1)  # [N,5]

        # Prepare RoIs for roi_align: [K, 5] (batch_idx_in_feat_map_tensor, x1, y1, x2, y2)
        roi_inputs = torch.cat([batch_indices_for_roi.unsqueeze(1).float(), rois_fm], dim=1)
        # roi_inputs2 = torch.cat([batch_indices_for_roi.unsqueeze(1).float(), rois_fm2], dim=1)
        
        if roi_inputs.numel() == 0: # Should be caught by bboxes_abs_img_scale.numel() == 0 earlier
             return torch.empty(0, self.feature_dim, device=device)
        # if roi_inputs2.numel() == 0: # Should be caught by bboxes_abs_img_scale.numel() == 0 earlier
        #      return torch.empty(0, self.feature_dim, device=device)
        
        # Perform RoI Align
        patches = roi_align(selected_feature_map, roi_inputs,
                            output_size=(self.roi_output_size, self.roi_output_size),
                            aligned=True) 
        # roi_aligned_patches shape: [N_roi, c_feat_map_for_roi, self.roi_output_size, self.roi_output_size]
        # patches2 = roi_align(selected_feature_map2, roi_inputs2,
        #                      output_size=(self.roi_output_size, self.roi_output_size),
        #                      aligned=True)
        # Encode the RoI-aligned patches
        encoded_features = encoder(patches)  # Expected output: [N_roi, self.feature_dim]
        #encoded_features2 = encoder(self.dim_map(patches2))  # Expected output: [N_roi, self.feature_dim]

        # L2 Normalize the encoded features
        #normalized_features = F.normalize(encoded_features, p=2, dim=1)
        #normalized_features2 = F.normalize(encoded_features2, p=2, dim=1)
        return encoded_features

    def forward(self, x_pyramid_from_neck: List[torch.Tensor], batch: dict = None):
        """
        Forward pass for DetectWithObjectMoCo.
        Args:
            x_pyramid_from_neck (List[torch.Tensor]): List of feature maps from FPN.
            batch (dict, optional): Batch data containing GT info, needed for MoCo during training.
                                    Expected keys: 'bboxes', 'cls', 'batch_idx', and image shape info.
        Returns:
            Tuple: During training: (det_head_outputs, raw_features_from_neck, 
                                     moco_query_features, moco_key_features, 
                                     moco_object_labels, moco_queue_snapshot)
                   During inference: (detection_results, raw_features_from_neck)
        """
        # Clone raw features from neck for MoCo, as x_pyramid_from_neck might be modified by detection heads
        raw_features_for_moco = [feat.clone() for feat in x_pyramid_from_neck]

        # Standard YOLO detection head processing
        det_head_outputs = []
        for i in range(self.nl):  # self.nl is number of detection layers
            det_head_outputs.append(torch.cat((self.cv2[i](x_pyramid_from_neck[i]), self.cv3[i](x_pyramid_from_neck[i])), 1))

        # --- MoCo Feature Extraction (only during training and if GT is available) ---
        moco_query_features = None
        moco_key_features = None
        moco_object_labels = None

        if self.training and batch is not None and 'bboxes' in batch and 'cls' in batch and 'batch_idx' in batch:
            gt_bboxes_img_scale = batch['bboxes']  # Expected [N_total_gt, 4] (xyxy absolute on input image)
            # Ensure labels are 1D: [N_total_gt]
            gt_labels = batch['cls'].view(-1) if batch['cls'].ndim > 1 else batch['cls']
            batch_indices_for_gt = batch['batch_idx'].view(-1) if batch['batch_idx'].ndim > 1 else batch['batch_idx']

            if gt_bboxes_img_scale.numel() > 0 and gt_labels.numel() > 0:
                # Get current batch image shapes [BatchSize, 2] (H, W)
                # This assumes batch['img'] is the augmented image tensor [B, C, H, W]
                current_batch_img_shapes = torch.tensor([batch['img'].shape[-2], batch['img'].shape[-1]],device=batch['img'].device,dtype=torch.float)
                # for i in range(batch['img'].shape[0]):
                #     h_img, w_img = batch['img'][i].shape[1], batch['img'][i].shape[2]
                #     current_batch_img_shapes_list.append(torch.tensor([h_img, w_img], device=batch['img'].device, dtype=torch.float))
                # current_batch_img_shapes = torch.stack(current_batch_img_shapes_list)

                # Extract Query Features
                moco_query_features = self._extract_roi_encoded_features(
                    raw_features_for_moco[0],
                    gt_bboxes_img_scale,
                    batch_indices_for_gt,
                    current_batch_img_shapes,
                    self.query_encoder
                )
                key_input_features = self.dim_map(raw_features_for_moco[1])  # Apply dim_map to the first feature map
                # Extract Key Features (with no_grad context for key_encoder path)
                with torch.no_grad():
                    self._momentum_update_key_encoder()  # Update key encoder parameters
                    moco_key_features = self._extract_roi_encoded_features(
                        key_input_features,
                        gt_bboxes_img_scale,
                        batch_indices_for_gt,
                        current_batch_img_shapes,
                        self.key_encoder
                    )
                
                moco_object_labels = gt_labels
                #moco_object_labels = torch.cat([gt_labels, gt_labels], dim=0)

                # Dequeue and Enqueue with detached key features
                if moco_key_features is not None and moco_key_features.numel() > 0:
                    self._dequeue_and_enqueue(moco_key_features.detach(), moco_object_labels)
        
        # --- Return appropriate outputs ---
        if self.training:
            # Pass all necessary components to the loss function
            return (det_head_outputs, raw_features_for_moco, 
                    moco_query_features, moco_key_features, 
                    moco_object_labels, self.queue.clone().detach()) # Pass a snapshot of the queue
        else:
            # Standard inference path
            # self._inference processes det_head_outputs to final detections
            y_det_inference = self._inference(det_head_outputs) 
            return y_det_inference if self.export else (y_det_inference, det_head_outputs)

# ... (rest of the file, e.g., _make_grid, _inference methods if not fully inherited)

class DetectWithMoCoBK(DetectWithObjectMoCo):
    """YOLO Detection head with MoCo and Backbone Knowledge distillation."""

    def __init__(self, nc=80, ch=(), queue_size=8192 * 2, momentum=0.999, feature_dim=32, roi_output_size=7):
        if len(ch) < 3:
            raise ValueError("DetectWithMoCoBK expects at least 3 channels: optional BK features followed by detect features.")

        bk_ch = ch[:2]
        det_ch = ch[2:]
        Detect.__init__(self, nc=nc, ch=det_ch)

        if len(bk_ch) < 2:
            raise ValueError("DetectWithMoCoBK expects two BK feature channels before the detect feature channels.")

        self.feature_dim = feature_dim
        self.roi_output_size = roi_output_size
        self.momentum = momentum
        self.queue_size = queue_size
        self.nc = nc
        self.hidden_dim = 2 * feature_dim

        c_feat_map_for_roi = bk_ch[0]
        feature_dim_2 = bk_ch[1]
        self.dim_map = nn.Sequential(
            nn.Conv2d(feature_dim_2, c_feat_map_for_roi, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_feat_map_for_roi),
            nn.ReLU()
        )
        self.query_encoder = EncoderWithProjectorEA(c_feat_map_for_roi, self.feature_dim, self.hidden_dim)
        self.key_encoder = EncoderWithProjectorEA(c_feat_map_for_roi, self.feature_dim, self.hidden_dim)

        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer('queue', torch.randn(self.nc, self.queue_size, self.feature_dim))
        self.queue = F.normalize(self.queue, p=2, dim=2)
        self.register_buffer('queue_ptr', torch.zeros(self.nc, dtype=torch.long))
        self.register_buffer('class_freq', torch.zeros(self.nc))
        self.total_samples = 0
    
    def forward(self, x_pyramid_from_neck: List[torch.Tensor], batch: dict = None):
        """
        Forward pass for DetectWithObjectMoCo.
        Args:
            x_pyramid_from_neck (List[torch.Tensor]): List of feature maps from FPN.
            batch (dict, optional): Batch data containing GT info, needed for MoCo during training.
                                    Expected keys: 'bboxes', 'cls', 'batch_idx', and image shape info.
        Returns:
            Tuple: During training: (det_head_outputs, raw_features_from_neck, 
                                     moco_query_features, moco_key_features, 
                                     moco_object_labels, moco_queue_snapshot)
                   During inference: (detection_results, raw_features_from_neck)
        """
        # Clone raw features from neck for MoCo, as x_pyramid_from_neck might be modified by detection heads
        raw_features_for_moco = [feat.clone() for feat in x_pyramid_from_neck[:2]]

        # Standard YOLO detection head processing
        det_features = x_pyramid_from_neck[2:]
        det_head_outputs = []
        for i in range(self.nl):  # self.nl is number of detection layers
            det_head_outputs.append(torch.cat((self.cv2[i](det_features[i]), self.cv3[i](det_features[i])), 1))

        # --- MoCo Feature Extraction (only during training and if GT is available) ---
        moco_query_features = None
        moco_key_features = None
        moco_object_labels = None

        if self.training and batch is not None and 'bboxes' in batch and 'cls' in batch and 'batch_idx' in batch:
            gt_bboxes_img_scale = batch['bboxes']  # Expected [N_total_gt, 4] (xyxy absolute on input image)
            # Ensure labels are 1D: [N_total_gt]
            gt_labels = batch['cls'].view(-1) if batch['cls'].ndim > 1 else batch['cls']
            batch_indices_for_gt = batch['batch_idx'].view(-1) if batch['batch_idx'].ndim > 1 else batch['batch_idx']

            if gt_bboxes_img_scale.numel() > 0 and gt_labels.numel() > 0:
                # Get current batch image shapes [BatchSize, 2] (H, W)
                # This assumes batch['img'] is the augmented image tensor [B, C, H, W]
                current_batch_img_shapes = torch.tensor([batch['img'].shape[-2], batch['img'].shape[-1]],device=batch['img'].device,dtype=torch.float)
                # for i in range(batch['img'].shape[0]):
                #     h_img, w_img = batch['img'][i].shape[1], batch['img'][i].shape[2]
                #     current_batch_img_shapes_list.append(torch.tensor([h_img, w_img], device=batch['img'].device, dtype=torch.float))
                # current_batch_img_shapes = torch.stack(current_batch_img_shapes_list)

                # Extract Query Features
                moco_query_features = self._extract_roi_encoded_features(
                    raw_features_for_moco[0],
                    gt_bboxes_img_scale,
                    batch_indices_for_gt,
                    current_batch_img_shapes,
                    self.query_encoder
                )
                key_input_features = self.dim_map(raw_features_for_moco[1])  # Apply dim_map to the first feature map
                # Extract Key Features (with no_grad context for key_encoder path)
                with torch.no_grad():
                    self._momentum_update_key_encoder()  # Update key encoder parameters
                    moco_key_features = self._extract_roi_encoded_features(
                        key_input_features,
                        gt_bboxes_img_scale,
                        batch_indices_for_gt,
                        current_batch_img_shapes,
                        self.key_encoder
                    )
                
                moco_object_labels = gt_labels
                #moco_object_labels = torch.cat([gt_labels, gt_labels], dim=0)

                # Dequeue and Enqueue with detached key features
                if moco_key_features is not None and moco_key_features.numel() > 0:
                    self._dequeue_and_enqueue(moco_key_features.detach(), moco_object_labels)
        
        # --- Return appropriate outputs ---
        if self.training:
            # Pass all necessary components to the loss function
            return (det_head_outputs, raw_features_for_moco, 
                    moco_query_features, moco_key_features, 
                    moco_object_labels, self.queue.clone().detach()) # Pass a snapshot of the queue
        else:
            # Standard inference path
            # self._inference processes det_head_outputs to final detections
            y_det_inference = self._inference(det_head_outputs) 
            return y_det_inference if self.export else (y_det_inference, det_head_outputs)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
