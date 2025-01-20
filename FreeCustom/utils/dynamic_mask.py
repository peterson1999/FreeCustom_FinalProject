import cv2
import numpy as np
import torch
from typing import List, Tuple

def compute_mask_shape_features(mask) :
    """
        area                   : 最大轮廓面积
        perimeter              : 最大轮廓周长
        solidity               : = area / hull_area, 轮廓实心度
        relative_area_ratio    : area / (H*W)
        perimeter_to_area_ratio: 周长 / 面积
    """
    mask_np = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
    h, w = mask_np.shape

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, True))

    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull)) if len(hull) > 2 else area
    if hull_area < 1e-6:
        solidity = 1.0
    else:
        solidity = area / hull_area

    relative_area_ratio = area / (h * w) if (h * w) > 0 else 0.0
    perimeter_to_area_ratio = perimeter / area if area > 1e-6 else 0.0
    return  solidity, relative_area_ratio, perimeter_to_area_ratio


def dynamic_mask_weights(
    masks, old_weights,
    alpha = 1.0,
    beta  = 1.0,
    gamma = 1.0,
    lambda_val=0.3
) :
    """  
    参数:
        masks : 一组掩码, 每个掩码形状 (1,1,H,W)
        alpha : 控制"面积比"的权重
        beta  : 控制"周长/面积比"的权重
        gamma : 控制"(1 - solidity)"的权重
        lambda_val : 控制原有权重和几何权重比例

    返回:
        weights: List[float], 与 masks 对应的一组浮点数，如 [3.0, 2.0, 2.1, ...]
    """
    weights = []
    for mask, w_old in zip(masks, old_weights):
        solidity, rel_area, p2a = compute_mask_shape_features(mask)
        
        if rel_area <= 0.1:
            rel_area = 0.1
        if p2a >= 0.5:
            p2a =0.5
        if solidity<=0.5:
            solidity=0.5
        # 形状越复杂、面积比越小、越不实心，则 weight 越高
        geom_factor = alpha * (1 - rel_area)   + beta  * p2a    + gamma * (1.0 - solidity)
        weight = (1 - lambda_val) * w_old + lambda_val * w_old * geom_factor
        
        weights.append(float(weight))

    
    return weights
 