import argparse
import json
import os
import shutil
import random
import logging
import pickle
from pathlib import Path
from threading import Thread
from collections import defaultdict

import numpy as np
import torch
import yaml
import cv2
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (coco80_to_coco91_class, check_dataset, check_file, check_img_size, 
                          check_requirements, box_iou, non_max_suppression, scale_coords, 
                          xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt, plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

# =============================================================================
# 1. 중복 라벨 제거 함수 추가 (라인 ~80 근처, safe_class_conversion 함수 뒤에 추가)
# =============================================================================

def remove_duplicate_labels(gt_boxes, tolerance=0.01):
    """
    중복 GT 라벨 제거 함수
    
    Args:
        gt_boxes: GT 라벨 리스트 [[cls, x, y, w, h], ...]
        tolerance: 중복 판정 허용 오차 (normalized coordinates)
    
    Returns:
        unique_boxes: 중복 제거된 라벨 리스트
        duplicate_info: 중복 정보 딕셔너리
    """
    if not gt_boxes:
        return [], {'removed_count': 0, 'duplicate_pairs': []}
    
    unique_boxes = []
    duplicate_pairs = []
    removed_indices = set()
    
    for i, box1 in enumerate(gt_boxes):
        if i in removed_indices:
            continue
            
        for j, box2 in enumerate(gt_boxes[i+1:], i+1):
            if j in removed_indices:
                continue
                
            try:
                # 같은 클래스인지 확인
                cls1 = safe_class_conversion(box1[0])
                cls2 = safe_class_conversion(box2[0])
                
                if cls1 == cls2 and len(box1) >= 5 and len(box2) >= 5:
                    # 위치 차이 계산 (normalized coordinates)
                    x_diff = abs(float(box1[1]) - float(box2[1]))
                    y_diff = abs(float(box1[2]) - float(box2[2]))
                    w_diff = abs(float(box1[3]) - float(box2[3]))
                    h_diff = abs(float(box1[4]) - float(box2[4]))
                    
                    # 중복 판정 (모든 좌표 차이가 tolerance 미만)
                    if (x_diff < tolerance and y_diff < tolerance and 
                        w_diff < tolerance and h_diff < tolerance):
                        
                        duplicate_pairs.append({
                            'kept_idx': i,
                            'removed_idx': j,
                            'differences': [x_diff, y_diff, w_diff, h_diff],
                            'class_id': cls1
                        })
                        removed_indices.add(j)  # 나중 인덱스 제거
                        
            except (ValueError, IndexError, TypeError):
                continue
        
        # 중복이 아닌 박스만 추가
        unique_boxes.append(box1)
    
    duplicate_info = {
        'removed_count': len(removed_indices),
        'duplicate_pairs': duplicate_pairs,
        'original_count': len(gt_boxes),
        'unique_count': len(unique_boxes)
    }
    
    return unique_boxes, duplicate_info


def improved_categorize_detection(precision, recall, conf_good, min_recall=0.8, min_precision=0.7):
    """
    개선된 검출 성능 카테고리 분류 함수
    
    Args:
        precision: 정밀도
        recall: 재현율 
        conf_good: 신뢰도 임계값 통과 여부
        min_recall: 최소 재현율 임계값
        min_precision: 최소 정밀도 임계값
    
    Returns:
        category: 분류된 카테고리명
    """
    
    # 1. 우수한 검출 (높은 정밀도 + 높은 재현율)
    if recall >= min_recall and precision >= min_precision:
        return 'good_detect' if conf_good else 'good_detect_low_conf'
    
    # 2. 부분적 성공 (매우 높은 정밀도 + 중간 재현율)
    elif precision >= 0.9 and recall >= 0.5:
        return 'partial_detect' if conf_good else 'partial_detect_low_conf'
    
    # 3. 경계선 성능 (중간 정밀도 + 중간 재현율)
    elif recall >= 0.4 and precision >= 0.6:
        return 'borderline_detect'
    
    # 4. 실제 검출 실패 (낮은 재현율)
    elif recall < 0.3:
        return 'miss_detect'
    
    # 5. 거짓 양성 문제 (낮은 정밀도)
    elif precision < 0.5:
        return 'false_detect'
    
    # 6. 기타 (주로 신뢰도 문제)
    else:
        return 'low_conf'


# 🛠 좌표 변환 문제 수정

# 🛠 coords_type 변수 선언 문제 수정

def evaluate_detection_integrated(pred, targets, img_path, names, conf_thres, iou_thres, 
                                img_shape, min_recall=0.8, classes=None, debug=False):
    """
    🎯 좌표 변환 문제를 해결한 통합 검출 평가 함수 (수정됨)
    """
    
    try:
        # 이미지 크기
        if img_shape and len(img_shape) >= 2:
            h, w = img_shape[0], img_shape[1]
        else:
            h, w = 640, 640
        
        if debug:
            print(f"\n🔍 COORDINATE FIXED DEBUG: {Path(img_path).name}")
            print(f"   Image size: {w}x{h}")
        
        # 🎯 coords_type 기본값 설정 (중요!)
        coords_type = "unknown"
        
        # 🎯 targets 상태 자동 감지 및 올바른 처리
        gt_boxes = []
        if len(targets) > 0:
            # 첫 번째 target으로 좌표 시스템 감지
            first_target = targets[0]
            if len(first_target) >= 5:
                x_test, y_test = first_target[1], first_target[2]
                
                # 좌표 시스템 자동 감지
                if 0 <= x_test <= 1 and 0 <= y_test <= 1:
                    coords_type = "normalized"
                    if debug:
                        print(f"   📍 Detected NORMALIZED coordinates")
                elif 0 <= x_test <= w and 0 <= y_test <= h:
                    coords_type = "pixel"  
                    if debug:
                        print(f"   📍 Detected PIXEL coordinates")
                else:
                    coords_type = "unknown"
                    if debug:
                        print(f"   ⚠️  Unknown coordinate system: x={x_test}, y={y_test}")
            
            for i, target in enumerate(targets):
                if len(target) >= 5:
                    if hasattr(target, 'cpu'):
                        gt_box = target.cpu().numpy()
                    else:
                        gt_box = np.array(target)
                    
                    cls_id = gt_box[0]
                    x_center, y_center, width, height = gt_box[1:5]
                    
                    # 🎯 좌표 시스템에 따른 적절한 처리
                    if coords_type == "normalized":
                        # 이미 normalized → 그대로 사용
                        normalized_gt = [cls_id, float(x_center), float(y_center), float(width), float(height)]
                    elif coords_type == "pixel":
                        # pixel → normalized 변환
                        norm_x = float(x_center) / w
                        norm_y = float(y_center) / h
                        norm_w = float(width) / w
                        norm_h = float(height) / h
                        normalized_gt = [cls_id, norm_x, norm_y, norm_w, norm_h]
                    else:
                        # 안전한 변환 시도
                        norm_x = float(x_center) / w if x_center > 1 else float(x_center)
                        norm_y = float(y_center) / h if y_center > 1 else float(y_center)
                        norm_w = float(width) / w if width > 1 else float(width)
                        norm_h = float(height) / h if height > 1 else float(height)
                        normalized_gt = [cls_id, norm_x, norm_y, norm_w, norm_h]
                    
                    gt_boxes.append(normalized_gt)
                    
                    # 🔍 상세 디버깅 정보
                    if debug and i < 3:
                        print(f"   GT {i+1}: cls={int(cls_id)}")
                        print(f"      Original: ({x_center:.6f}, {y_center:.6f}, {width:.6f}, {height:.6f})")
                        print(f"      Normalized: ({normalized_gt[1]:.6f}, {normalized_gt[2]:.6f}, {normalized_gt[3]:.6f}, {normalized_gt[4]:.6f})")
                        
                        # 좌표 범위 검증
                        if not (0 <= normalized_gt[1] <= 1 and 0 <= normalized_gt[2] <= 1):
                            print(f"      ⚠️  WARNING: Normalized coordinates out of range!")
        else:
            if debug:
                print(f"   📍 No targets found - coords_type set to 'unknown'")

        # 🎯 중복 라벨 제거 (고정밀도 tolerance 사용)
        original_gt_count = len(gt_boxes)
        if gt_boxes:
            # 좌표 정밀도를 고려한 tolerance 조정
            high_precision_tolerance = 0.001  # 더 작은 tolerance 사용
            unique_gt_boxes, duplicate_info = remove_duplicate_labels(gt_boxes, high_precision_tolerance)
            if debug and duplicate_info['removed_count'] > 0:
                print(f"   🔧 고정밀도 중복 제거: {original_gt_count} → {len(unique_gt_boxes)} GT")
                for dup in duplicate_info['duplicate_pairs']:
                    diffs = dup['differences']
                    print(f"      제거: GT{dup['removed_idx']} (GT{dup['kept_idx']}와 중복)")
                    print(f"         차이: x={diffs[0]:.6f}, y={diffs[1]:.6f}, w={diffs[2]:.6f}, h={diffs[3]:.6f}")
        else:
            unique_gt_boxes = gt_boxes
            duplicate_info = {'removed_count': 0, 'duplicate_pairs': []}
        
        # 클래스 필터링 (unique GT 사용)
        filtered_gt_boxes = []
        if classes is not None:
            for gt_idx, gt_box in enumerate(unique_gt_boxes):
                gt_cls = safe_class_conversion(gt_box[0])
                if gt_cls < len(names) and gt_cls in [int(cls) for cls in classes]:
                    filtered_gt_boxes.append((gt_idx, gt_box))
        else:
            for gt_idx, gt_box in enumerate(unique_gt_boxes):
                gt_cls = safe_class_conversion(gt_box[0])
                if gt_cls < len(names):
                    filtered_gt_boxes.append((gt_idx, gt_box))
        
        if debug:
            print(f"   GT: {original_gt_count} total → {len(unique_gt_boxes)} unique → {len(filtered_gt_boxes)} filtered")
        
        # ✅ Prediction 처리 (기존과 동일하지만 고정밀도)
        pred_boxes = []
        filtered_det = []
        
        if pred is not None and len(pred) > 0:
            if debug:
                print(f"   Pred: {len(pred)} detections")
            
            for i, (*xyxy, conf, cls) in enumerate(pred):
                # PIXEL → NORMALIZED 변환 (고정밀도)
                x1, y1, x2, y2 = xyxy
                x_center = ((float(x1) + float(x2)) / 2) / w
                y_center = ((float(y1) + float(y2)) / 2) / h
                width = (float(x2) - float(x1)) / w
                height = (float(y2) - float(y1)) / h
                
                # 디버깅: 변환된 좌표 범위 체크
                if debug and i < 3:
                    print(f"   Pred {i+1}: cls={int(cls)}, conf={conf:.3f}")
                    print(f"      pixel=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                    print(f"      norm=({x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f})")
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        print(f"      ⚠️  WARNING: Pred coordinates out of range!")
                
                # 저장할 정보 구성 (고정밀도)
                pred_info = {
                    'pixel_coords': [float(x1), float(y1), float(x2), float(y2)],
                    'normalized_coords': [x_center, y_center, width, height],  # 고정밀도 유지
                    'confidence': float(conf),
                    'class_id': int(cls)
                }
                pred_boxes.append(pred_info)
            
            # 클래스 필터링
            if classes is not None:
                for i, pred_info in enumerate(pred_boxes):
                    cls_id = pred_info['class_id']
                    if cls_id < len(names) and cls_id in [int(cls) for cls in classes]:
                        filtered_det.append((i, pred_info))
            else:
                for i, pred_info in enumerate(pred_boxes):
                    cls_id = pred_info['class_id']
                    if cls_id < len(names):
                        filtered_det.append((i, pred_info))
            
            if debug:
                print(f"   Filtered pred: {len(filtered_det)}")
        
        # 신뢰도 체크
        all_conf_good = True
        if filtered_det:
            all_conf_good = all(pred_info['confidence'] >= conf_thres for _, pred_info in filtered_det)
        
        # ✅ IoU 매칭 (고정밀도 좌표 사용)
        matched_gt = set()
        matched_pred = set()
        match_details = []
        
        for pred_idx, pred_info in filtered_det:
            pred_cls = pred_info['class_id']
            if pred_cls >= len(names):
                continue
            
            pred_bbox = pred_info['normalized_coords']
            
            best_iou = 0
            best_gt_idx = -1
            best_gt_original_idx = -1
            
            for orig_gt_idx, gt_box in filtered_gt_boxes:
                gt_cls = safe_class_conversion(gt_box[0])
                if gt_cls >= len(names) or pred_cls != gt_cls:
                    continue
                
                gt_bbox = gt_box[1:5]
                iou = calculate_bbox_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = len(match_details)
                    best_gt_original_idx = orig_gt_idx
            
            if best_iou > iou_thres and best_gt_original_idx >= 0:
                matched_gt.add(best_gt_original_idx)
                matched_pred.add(pred_idx)
                
                match_details.append({
                    'pred_idx': pred_idx,
                    'gt_idx': best_gt_original_idx,
                    'iou': float(best_iou),
                    'pred_conf': pred_info['confidence'],
                    'class_id': pred_cls
                })
                
                # 디버깅: 매칭 정보 출력
                if debug:
                    print(f"   ✅ HIGH-PRECISION MATCH: Pred {pred_idx} ↔ GT {best_gt_original_idx}, IoU={best_iou:.4f}, conf={pred_info['confidence']:.3f}")
        
        # ✅ 메트릭 계산 (unique GT 기준)
        filtered_gt_count = len(filtered_gt_boxes)
        filtered_pred_count = len(filtered_det)
        
        if filtered_pred_count > 0:
            precision = len(matched_pred) / filtered_pred_count
        else:
            precision = 1.0 if filtered_gt_count == 0 else 0.0
            
        if filtered_gt_count > 0:
            recall = len(matched_gt) / filtered_gt_count
        else:
            recall = 1.0 if filtered_pred_count == 0 else 0.0
        
        # 🎯 개선된 카테고리 결정
        if filtered_gt_count > 0:
            if recall >= min_recall and precision >= 0.7:
                category = 'good_detect' if all_conf_good else 'good_detect_low_conf'
            elif precision >= 0.9 and recall >= 0.5:
                category = 'partial_detect' if all_conf_good else 'partial_detect_low_conf'
            elif recall >= 0.4 and precision >= 0.6:
                category = 'borderline_detect'
            elif recall < 0.3:
                category = 'miss_detect'
            elif precision < 0.5:
                category = 'false_detect'
            else:
                category = 'low_conf'
        else:
            category = 'false_detect' if filtered_pred_count > 0 else 'background'
        
        # 🎯 고정밀도 디버깅 정보
        if debug:
            print(f"   📊 HIGH-PRECISION RESULT: {category}")
            print(f"      Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"      Coordinate system: {coords_type}")
            print(f"      Duplicates: {duplicate_info['removed_count']} removed (tolerance: 0.001)")
            print(f"      Matched: GT={len(matched_gt)}/{filtered_gt_count}, Pred={len(matched_pred)}/{filtered_pred_count}")
        
        # 매칭 정보 및 메트릭 정보 구성
        matched_info = {
            'matched_gt': list(matched_gt),
            'matched_pred': list(matched_pred),
            'match_details': match_details
        }
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'gt_count': original_gt_count,
            'unique_gt_count': len(unique_gt_boxes),
            'pred_count': len(pred_boxes),
            'filtered_gt_count': filtered_gt_count,
            'filtered_pred_count': filtered_pred_count,
            'matched_gt_count': len(matched_gt),
            'matched_pred_count': len(matched_pred),
            'duplicate_removed_count': duplicate_info['removed_count'],
            'confidence_good': all_conf_good,
            'coordinate_system': coords_type,  # ✅ 이제 항상 선언됨
            'category_reason': f"P={precision:.4f}, R={recall:.4f}, coord={coords_type}, dup_removed={duplicate_info['removed_count']}"
        }
        
        return DetectionResult(
            img_path=img_path,
            category=category,
            pred_info=pred_boxes,
            gt_info=unique_gt_boxes,
            metrics=metrics,
            matched_info=matched_info
        )
        
    except Exception as e:
        print(f"❌ ERROR in coordinate-fixed evaluation for {img_path}: {e}")
        import traceback
        traceback.print_exc()
        
        metrics = {
            'precision': 0.0, 'recall': 0.0, 'gt_count': 0, 'pred_count': 0,
            'filtered_gt_count': 0, 'filtered_pred_count': 0,
            'matched_gt_count': 0, 'matched_pred_count': 0, 'error': str(e)
        }
        
        return DetectionResult(
            img_path=img_path,
            category='background',
            pred_info=[],
            gt_info=[],
            metrics=metrics
        )
def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def safe_class_conversion(value):
    try:
        return int(float(value))  # float()로 먼저 변환 후 int()
    except (ValueError, TypeError):
        return 0  # 변환 실패 시 기본값
    
class DetectionResult:
    """JSON 직렬화 가능한 검출 결과 저장 클래스"""
    def __init__(self, img_path, category, pred_info, gt_info, metrics, matched_info=None):
        self.img_path = str(img_path)
        self.category = category
        self.pred_info = pred_info
        self.gt_info = gt_info
        self.metrics = metrics
        self.matched_info = matched_info or {'matched_gt': [], 'matched_pred': []}
        self.parent_name = self.get_parent_name()
    
    def get_parent_name(self):
        """부모 디렉토리 이름 추출"""
        try:
            path_obj = Path(self.img_path)
            path_parts = path_obj.parts
            
            if 'JPEGImages' in path_parts:
                jpeg_idx = path_parts.index('JPEGImages')
                if jpeg_idx > 0:
                    return path_parts[jpeg_idx - 1]
            elif 'valid' in path_parts:
                valid_idx = path_parts.index('valid')
                if valid_idx > 0:
                    return path_parts[valid_idx - 1]
            elif 'images' in path_parts:
                images_idx = path_parts.index('images')
                if images_idx > 0:
                    return path_parts[images_idx - 1]
            
            if len(path_parts) >= 3:
                return path_parts[-3]
            elif len(path_parts) >= 2:
                return path_parts[-2]
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def to_dict(self):
        """JSON 직렬화 가능한 딕셔너리로 변환"""
        return {
            'img_path': self.img_path,
            'category': self.category,
            'pred_info': self.pred_info,
            'gt_info': self.gt_info,
            'metrics': self.metrics,
            'matched_info': self.matched_info,
            'parent_name': self.parent_name
        }


def calculate_bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes (normalized coordinates)."""
    try:
        def xywh_to_xyxy(box):
            x_center, y_center, width, height = box[:4]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_xyxy = xywh_to_xyxy(box1)
        box2_xyxy = xywh_to_xyxy(box2)
        
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    except:
        return 0.0


# 🛠 evaluate_detection_integrated 함수 완전 수정

# def evaluate_detection_integrated(pred, targets, img_path, names, conf_thres, iou_thres, 
#                                 img_shape, min_recall=0.8, classes=None, debug=False):
#     """
#     🎯 통합 완성 버전: 중복 라벨 제거 + 좌표 수정 + 메모리 효율성
#     """
    
#     try:
#         # 이미지 크기
#         if img_shape and len(img_shape) >= 2:
#             h, w = img_shape[0], img_shape[1]
#         else:
#             h, w = 640, 640
        
#         if debug:
#             print(f"\n🔍 FIXED DEBUG: {Path(img_path).name}")
#             print(f"   Image size: {w}x{h}")
        
#         # ✅ Ground Truth 처리 (targets는 pixel 좌표 → normalized로 변환)
#         gt_boxes = []
#         if len(targets) > 0:
#             for target in targets:
#                 if len(target) >= 5:
#                     if hasattr(target, 'cpu'):
#                         gt_box = target.cpu().numpy()
#                     else:
#                         gt_box = np.array(target)
                    
#                     # pixel 좌표를 normalized 좌표로 변환
#                     cls_id = gt_box[0]
#                     x_center_pixel, y_center_pixel, width_pixel, height_pixel = gt_box[1:5]
                    
#                     # normalized 변환
#                     x_center = x_center_pixel / w
#                     y_center = y_center_pixel / h
#                     width = width_pixel / w
#                     height = height_pixel / h
                    
#                     normalized_gt = [cls_id, x_center, y_center, width, height]
#                     gt_boxes.append(normalized_gt)
                    
#                     # 디버깅: GT 좌표 범위 체크
#                     if debug and len(gt_boxes) <= 3:
#                         print(f"   GT {len(gt_boxes)}: cls={int(cls_id)}")
#                         print(f"      pixel=({x_center_pixel:.1f},{y_center_pixel:.1f},{width_pixel:.1f},{height_pixel:.1f})")
#                         print(f"      norm=({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})")
#                         if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
#                             print(f"   ⚠️  WARNING: GT coordinates out of normalized range!")

#         # 🎯 중복 라벨 제거 (모든 GT 수집 후 한 번만 실행)
#         original_gt_count = len(gt_boxes)
#         if gt_boxes:
#             unique_gt_boxes, duplicate_info = remove_duplicate_labels(gt_boxes, 0.01)
#             if debug and duplicate_info['removed_count'] > 0:
#                 print(f"   🔧 중복 제거: {original_gt_count} → {len(unique_gt_boxes)} GT")
#                 for dup in duplicate_info['duplicate_pairs']:
#                     print(f"      제거: GT{dup['removed_idx']} (GT{dup['kept_idx']}와 중복)")
#         else:
#             unique_gt_boxes = gt_boxes
#             duplicate_info = {'removed_count': 0, 'duplicate_pairs': []}
        
#         # 클래스 필터링된 GT 박스 (unique_gt_boxes 사용)
#         filtered_gt_boxes = []
#         if classes is not None:
#             for gt_idx, gt_box in enumerate(unique_gt_boxes):
#                 gt_cls = safe_class_conversion(gt_box[0])
#                 if gt_cls < len(names) and gt_cls in [int(cls) for cls in classes]:
#                     filtered_gt_boxes.append((gt_idx, gt_box))
#         else:
#             for gt_idx, gt_box in enumerate(unique_gt_boxes):
#                 gt_cls = safe_class_conversion(gt_box[0])
#                 if gt_cls < len(names):
#                     filtered_gt_boxes.append((gt_idx, gt_box))
        
#         if debug:
#             print(f"   GT: {original_gt_count} total → {len(unique_gt_boxes)} unique → {len(filtered_gt_boxes)} filtered")
        
#         # ✅ Prediction 처리 (PIXEL 좌표 → NORMALIZED 변환)
#         pred_boxes = []
#         filtered_det = []
        
#         if pred is not None and len(pred) > 0:
#             if debug:
#                 print(f"   Pred: {len(pred)} detections")
            
#             for i, (*xyxy, conf, cls) in enumerate(pred):
#                 # PIXEL 좌표를 NORMALIZED 좌표로 변환
#                 x1, y1, x2, y2 = xyxy
#                 x_center = ((x1 + x2) / 2) / w
#                 y_center = ((y1 + y2) / 2) / h
#                 width = (x2 - x1) / w
#                 height = (y2 - y1) / h
                
#                 # 디버깅: 변환된 좌표 범위 체크
#                 if debug and i < 3:
#                     print(f"   Pred {i+1}: cls={int(cls)}, conf={conf:.3f}")
#                     print(f"      pixel=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
#                     print(f"      norm=({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})")
#                     if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
#                         print(f"   ⚠️  WARNING: Pred coordinates out of range!")
                
#                 # 저장할 정보 구성 (JSON 직렬화 안전)
#                 pred_info = {
#                     'pixel_coords': [float(x1), float(y1), float(x2), float(y2)],
#                     'normalized_coords': [float(x_center), float(y_center), float(width), float(height)],
#                     'confidence': float(conf),
#                     'class_id': int(cls)
#                 }
#                 pred_boxes.append(pred_info)
            
#             # 클래스 필터링
#             if classes is not None:
#                 for i, pred_info in enumerate(pred_boxes):
#                     cls_id = pred_info['class_id']
#                     if cls_id < len(names) and cls_id in [int(cls) for cls in classes]:
#                         filtered_det.append((i, pred_info))
#             else:
#                 for i, pred_info in enumerate(pred_boxes):
#                     cls_id = pred_info['class_id']
#                     if cls_id < len(names):
#                         filtered_det.append((i, pred_info))
            
#             if debug:
#                 print(f"   Filtered pred: {len(filtered_det)}")
        
#         # 신뢰도 체크
#         all_conf_good = True
#         if filtered_det:
#             all_conf_good = all(pred_info['confidence'] >= conf_thres for _, pred_info in filtered_det)
        
#         # ✅ IoU 매칭 (모두 NORMALIZED 좌표 사용)
#         matched_gt = set()
#         matched_pred = set()
#         match_details = []
        
#         for pred_idx, pred_info in filtered_det:
#             pred_cls = pred_info['class_id']
#             if pred_cls >= len(names):
#                 continue
            
#             pred_bbox = pred_info['normalized_coords']
            
#             best_iou = 0
#             best_gt_idx = -1
#             best_gt_original_idx = -1
            
#             for orig_gt_idx, gt_box in filtered_gt_boxes:
#                 gt_cls = safe_class_conversion(gt_box[0])
#                 if gt_cls >= len(names) or pred_cls != gt_cls:
#                     continue
                
#                 gt_bbox = gt_box[1:5]
#                 iou = calculate_bbox_iou(pred_bbox, gt_bbox)
                
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_gt_idx = len(match_details)  # 매치 리스트에서의 인덱스
#                     best_gt_original_idx = orig_gt_idx
            
#             if best_iou > iou_thres and best_gt_original_idx >= 0:
#                 matched_gt.add(best_gt_original_idx)
#                 matched_pred.add(pred_idx)
                
#                 match_details.append({
#                     'pred_idx': pred_idx,
#                     'gt_idx': best_gt_original_idx,
#                     'iou': float(best_iou),
#                     'pred_conf': pred_info['confidence'],
#                     'class_id': pred_cls
#                 })
                
#                 # 디버깅: 매칭 정보 출력
#                 if debug:
#                     print(f"   ✅ MATCH: Pred {pred_idx} ↔ GT {best_gt_original_idx}, IoU={best_iou:.3f}, conf={pred_info['confidence']:.3f}")
        
#         # ✅ 메트릭 계산 (unique GT 기준)
#         filtered_gt_count = len(filtered_gt_boxes)
#         filtered_pred_count = len(filtered_det)
        
#         if filtered_pred_count > 0:
#             precision = len(matched_pred) / filtered_pred_count
#         else:
#             precision = 1.0 if filtered_gt_count == 0 else 0.0
            
#         if filtered_gt_count > 0:
#             recall = len(matched_gt) / filtered_gt_count
#         else:
#             recall = 1.0 if filtered_pred_count == 0 else 0.0
        
#         # 🎯 개선된 카테고리 결정
#         if filtered_gt_count > 0:
#             # 1. 우수한 검출 (높은 정밀도 + 높은 재현율)
#             if recall >= min_recall and precision >= 0.7:
#                 category = 'good_detect' if all_conf_good else 'good_detect_low_conf'
#             # 2. 부분적 성공 (매우 높은 정밀도 + 중간 재현율)
#             elif precision >= 0.9 and recall >= 0.5:
#                 category = 'partial_detect' if all_conf_good else 'partial_detect_low_conf'
#             # 3. 경계선 성능 (중간 정밀도 + 중간 재현율)
#             elif recall >= 0.4 and precision >= 0.6:
#                 category = 'borderline_detect'
#             # 4. 실제 검출 실패 (낮은 재현율)
#             elif recall < 0.3:
#                 category = 'miss_detect'
#             # 5. 거짓 양성 문제 (낮은 정밀도)
#             elif precision < 0.5:
#                 category = 'false_detect'
#             # 6. 기타 (주로 신뢰도 문제)
#             else:
#                 category = 'low_conf'
#         else:
#             category = 'false_detect' if filtered_pred_count > 0 else 'background'
        
#         # 🎯 개선된 디버깅 정보
#         if debug:
#             print(f"   📊 FIXED RESULT: {category}")
#             print(f"      Precision: {precision:.3f}, Recall: {recall:.3f}")
#             print(f"      Duplicates: {duplicate_info['removed_count']} removed")
#             print(f"      Original GT: {original_gt_count} → Unique: {len(unique_gt_boxes)}")
#             print(f"      Matched: GT={len(matched_gt)}/{filtered_gt_count}, Pred={len(matched_pred)}/{filtered_pred_count}")
        
#         # 매칭 정보 (JSON 직렬화 안전)
#         matched_info = {
#             'matched_gt': list(matched_gt),
#             'matched_pred': list(matched_pred),
#             'match_details': match_details
#         }
        
#         # 🎯 확장된 메트릭 정보 (중복 정보 포함)
#         metrics = {
#             'precision': float(precision),
#             'recall': float(recall),
#             'gt_count': original_gt_count,  # 원본 GT 수
#             'unique_gt_count': len(unique_gt_boxes),  # 중복 제거 후 GT 수
#             'pred_count': len(pred_boxes),
#             'filtered_gt_count': filtered_gt_count,
#             'filtered_pred_count': filtered_pred_count,
#             'matched_gt_count': len(matched_gt),
#             'matched_pred_count': len(matched_pred),
#             'duplicate_removed_count': duplicate_info['removed_count'],  # 중복 제거 수
#             'confidence_good': all_conf_good,
#             'category_reason': f"P={precision:.3f}, R={recall:.3f}, dup_removed={duplicate_info['removed_count']}"
#         }
        
#         return DetectionResult(
#             img_path=img_path,
#             category=category,
#             pred_info=pred_boxes,
#             gt_info=unique_gt_boxes,  # 🎯 중복 제거된 GT 저장
#             metrics=metrics,
#             matched_info=matched_info
#         )
        
#     except Exception as e:
#         print(f"❌ ERROR in fixed evaluation for {img_path}: {e}")
#         # 오류 시 기본 결과 반환
#         metrics = {
#             'precision': 0.0, 'recall': 0.0, 'gt_count': 0, 'pred_count': 0,
#             'filtered_gt_count': 0, 'filtered_pred_count': 0,
#             'matched_gt_count': 0, 'matched_pred_count': 0, 'error': str(e)
#         }
        
#         return DetectionResult(
#             img_path=img_path,
#             category='background',
#             pred_info=[],
#             gt_info=[],
#             metrics=metrics
#         )


def create_visualization_integrated(result_data, names, conf_thres, iou_thres):
    """
    🎯 통합 시각화 함수 - 매칭 정보 포함
    """
    try:
        # 결과 데이터 추출
        if isinstance(result_data, dict):
            img_path = result_data['img_path']
            category = result_data['category']
            pred_info = result_data['pred_info']
            gt_info = result_data['gt_info']
            metrics = result_data['metrics']
            matched_info = result_data.get('matched_info', {'matched_gt': [], 'matched_pred': []})
        else:
            img_path = result_data.img_path
            category = result_data.category
            pred_info = result_data.pred_info
            gt_info = result_data.gt_info
            metrics = result_data.metrics
            matched_info = result_data.matched_info
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Image not found: {Path(img_path).name}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        h, w = img.shape[:2]
        vis_img = img.copy()
        
        # 색상 정의
        gt_color = (0, 255, 0)       # 초록: GT (unmatched)
        pred_color = (0, 0, 255)     # 빨강: Prediction (unmatched)
        matched_color = (255, 0, 0)  # 파랑: 매치됨
        low_conf_color = (128, 128, 128)  # 회색: 낮은 신뢰도
        
        matched_gt_indices = set(matched_info.get('matched_gt', []))
        matched_pred_indices = set(matched_info.get('matched_pred', []))
        
        # ✅ Ground Truth 박스 그리기 (NORMALIZED → PIXEL)
        for gt_idx, gt_box in enumerate(gt_info):
            if len(gt_box) >= 5:
                cls_id = safe_class_conversion(gt_box[0])  # ✅ '0.0' → 0
                if cls_id < len(names):
                    # 정규화된 좌표를 픽셀 좌표로 변환
                    x_center, y_center, width, height = gt_box[1:5]
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # 경계 체크
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # 매칭 여부에 따른 색상 선택
                    color = matched_color if gt_idx in matched_gt_indices else gt_color
                    thickness = 4 if gt_idx in matched_gt_indices else 3
                    
                    # GT 박스 그리기
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
                    
                    # 라벨 추가
                    label = f"GT: {names.get(cls_id, f'class_{cls_id}')}"
                    if gt_idx in matched_gt_indices:
                        label += " (Matched)"
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_img, (x1, y1-35), (x1+label_size[0]+10, y1), color, -1)
                    cv2.putText(vis_img, label, (x1+5, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ✅ Prediction 박스 그리기
        for pred_idx, pred_box in enumerate(pred_info):
            if isinstance(pred_box, dict):
                # 픽셀 좌표 직접 사용
                if 'pixel_coords' in pred_box:
                    x1, y1, x2, y2 = pred_box['pixel_coords']
                else:
                    # 정규화된 좌표에서 변환
                    x_center, y_center, width, height = pred_box['normalized_coords']
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                
                conf = pred_box['confidence']
                cls_id = pred_box['class_id']
            else:
                # 기존 형식 호환성
                if len(pred_box) >= 6:
                    x1, y1, x2, y2, conf, cls_id = pred_box[:6]
                    cls_id = safe_class_conversion(gt_box[0])
                else:
                    continue
            
            if cls_id < len(names):
                # 매칭 여부와 신뢰도에 따른 색상 선택
                if pred_idx in matched_pred_indices:
                    color = matched_color
                    thickness = 4
                elif conf >= conf_thres:
                    color = pred_color
                    thickness = 3
                else:
                    color = low_conf_color
                    thickness = 2
                
                # 경계 체크
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                # Prediction 박스 그리기
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨 추가
                label = f"Pred: {names.get(cls_id, f'class_{cls_id}')} {conf:.2f}"
                if pred_idx in matched_pred_indices:
                    label += " (Matched)"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_img, (x1, y2), (x1+label_size[0]+10, y2+35), color, -1)
                cv2.putText(vis_img, label, (x1+5, y2+25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 정보 패널 추가
        info_panel_height = 180
        info_panel = np.zeros((info_panel_height, w, 3), dtype=np.uint8)
        
        info_texts = [
            f"Category: {category}",
            f"Precision: {metrics.get('precision', 0):.3f}, Recall: {metrics.get('recall', 0):.3f}",
            f"GT Total: {metrics.get('gt_count', 0)}, Filtered: {metrics.get('filtered_gt_count', 0)}",
            f"Pred Total: {metrics.get('pred_count', 0)}, Filtered: {metrics.get('filtered_pred_count', 0)}",
            f"Matched GT: {metrics.get('matched_gt_count', 0)}, Matched Pred: {metrics.get('matched_pred_count', 0)}",
            f"Confidence Threshold: {conf_thres}, IoU Threshold: {iou_thres}",
            f"Image: {Path(img_path).name} ({w}x{h})"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(info_panel, text, (10, 25 + i*22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 범례 추가
        legend_x = w - 300
        legend_y = 25
        
        cv2.rectangle(info_panel, (legend_x, legend_y), (legend_x+20, legend_y+15), gt_color, -1)
        cv2.putText(info_panel, "GT (Unmatched)", (legend_x+25, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(info_panel, (legend_x, legend_y+25), (legend_x+20, legend_y+40), pred_color, -1)
        cv2.putText(info_panel, "Pred (Unmatched)", (legend_x+25, legend_y+37), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(info_panel, (legend_x, legend_y+50), (legend_x+20, legend_y+65), matched_color, -1)
        cv2.putText(info_panel, "Matched", (legend_x+25, legend_y+62), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(info_panel, (legend_x, legend_y+75), (legend_x+20, legend_y+90), low_conf_color, -1)
        cv2.putText(info_panel, "Low Confidence", (legend_x+25, legend_y+87), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 최종 이미지 합성
        final_img = np.vstack([vis_img, info_panel])
        
        return final_img
        
    except Exception as e:
        print(f"Error creating integrated visualization: {e}")
        dummy_img = np.zeros((400, 800, 3), dtype=np.uint8)
        cv2.putText(dummy_img, f"Visualization Error: {str(e)}", (10, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return dummy_img


def post_process_integrated(results_file, save_dir, names, conf_thres, iou_thres, 
                          max_images_per_category=1000):
    """통합 후처리 함수"""
    
    logger = setup_logging()
    
    try:
        # 결과 파일 로드
        logger.info(f"Loading results from {results_file}...")
        
        with open(results_file, 'r') as f:
            all_results = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(all_results)} results")
        
        # false/miss detection만 필터링
        problem_results = []
        for result in all_results:
            category = result.get('category', 'background')
            if category in ['false_detect', 'miss_detect', 'low_conf', 'partial_detect', 'borderline_detect']:  # 🎯 카테고리 추가
                problem_results.append(result)
        
        logger.info(f"Found {len(problem_results)} problem detections")
        
        # 카테고리별 분류
        categorized_results = defaultdict(list)
        for result in problem_results:
            category = result.get('category', 'background')
            categorized_results[category].append(result)
        
        # 각 카테고리별 처리
        for category, results in categorized_results.items():
            logger.info(f"Processing {category}: {len(results)} images")
            
            # 디렉토리 생성
            category_dir = save_dir / 'categorized_results' / 'overall' / category
            category_dir.mkdir(parents=True, exist_ok=True)
            (category_dir / 'JPEGImages').mkdir(exist_ok=True)
            (category_dir / 'labels').mkdir(exist_ok=True)
            (category_dir / 'debug_images').mkdir(exist_ok=True)
            (category_dir / 'metadata').mkdir(exist_ok=True)
            
            # 최대 개수 제한
            if len(results) > max_images_per_category:
                results = random.sample(results, max_images_per_category)
                logger.info(f"Sampled {max_images_per_category} images from {category}")
            
            # 각 이미지 처리
            for i, result in enumerate(tqdm(results, desc=f"Processing {category}")):
                try:
                    img_path = result['img_path']
                    gt_info = result['gt_info']
                    
                    img_name = Path(img_path).name
                    img_stem = Path(img_path).stem
                    
                    # 1. 원본 이미지 복사
                    if os.path.exists(img_path):
                        shutil.copy(img_path, category_dir / 'JPEGImages' / img_name)
                    
                    # 2. 라벨 파일 생성 (normalized 좌표)
                    if gt_info:
                        label_content = []
                        for gt_box in gt_info:
                            if len(gt_box) >= 5:
                                cls_id = int(float(gt_box[0]))
                                x, y, w, h = gt_box[1:5]
                                label_content.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                        
                        if label_content:
                            label_path = category_dir / 'labels' / f"{img_stem}.txt"
                            with open(label_path, 'w') as f:
                                f.write('\n'.join(label_content))
                    
                    # 3. 시각화 이미지 생성
                    vis_img = create_visualization_integrated(result, names, conf_thres, iou_thres)
                    vis_path = category_dir / 'debug_images' / f"{img_stem}_analysis.jpg"
                    cv2.imwrite(str(vis_path), vis_img)
                    
                    # 4. 메타데이터 저장
                    metadata_path = category_dir / 'metadata' / f"{img_stem}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(result, f, indent=2)
                
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
        
        logger.info("Integrated post-processing completed successfully")
        
        # 요약 정보 생성
        generate_integrated_summary(categorized_results, save_dir)
        
    except Exception as e:
        logger.error(f"Error in integrated post-processing: {e}")
        raise


def generate_integrated_summary(categorized_results, save_dir):
    """통합 요약 보고서 생성"""
    logger = setup_logging()
    
    try:
        total_problems = sum(len(results) for results in categorized_results.values())
        
        # 텍스트 요약 생성
        summary_content = [
            "🎯 === INTEGRATED FALSE/MISS DETECTION ANALYSIS ===",
            f"📊 Total problem detections: {total_problems}",
            f"🕐 Analysis completed: {Path().cwd()}",
            ""
        ]
        
        for category, results in categorized_results.items():
            count = len(results)
            percentage = (count / total_problems * 100) if total_problems > 0 else 0
            summary_content.append(f"📂 {category}: {count} ({percentage:.1f}%)")
        
        summary_content.extend([
            "",
            "🔍 === ANALYSIS LOCATIONS ===",
            "📁 Original images: categorized_results/overall/{category}/JPEGImages/",
            "🏷️  Labels: categorized_results/overall/{category}/labels/",
            "🎨 Visualizations: categorized_results/overall/{category}/debug_images/",
            "📋 Metadata: categorized_results/overall/{category}/metadata/",
            "",
            "💡 === RECOMMENDED ACTIONS ===",
            "🔴 1. Review false_detect images for background patterns",
            "🟡 2. Analyze miss_detect images for small objects or occlusions", 
            "🟠 3. Check low_conf images for threshold optimization",
            "📈 4. Focus on categories with highest percentages",
            "",
            "🎯 === NEXT STEPS ===",
            "• Use debug_images/ for visual pattern analysis",
            "• Check metadata/ for detailed metrics",
            "• Compare with baseline performance",
            "• Plan targeted data augmentation"
        ])
        
        summary_path = save_dir / 'integrated_analysis_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_content))
        
        logger.info(f"📄 Integrated summary saved to {summary_path}")
        
        # JSON 통계 저장
        stats = {
            'total_problems': total_problems,
            'by_category': {cat: len(results) for cat, results in categorized_results.items()},
            'analysis_type': 'integrated_coordinate_fixed',
            'analysis_completed': True,
            'recommendations': {
                'false_detect': 'Add hard negative mining and background data',
                'miss_detect': 'Increase small object detection and reduce occlusion',
                'low_conf': 'Optimize confidence threshold and model training'
            }
        }
        
        stats_path = save_dir / 'integrated_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Integrated statistics saved to {stats_path}")
        
    except Exception as e:
        logger.error(f"Error generating integrated summary: {e}")


def test_final_integrated(data,
                        weights=None,
                        batch_size=32,
                        imgsz=640,
                        conf_thres=0.001,
                        iou_thres=0.6,
                        save_json=False,
                        single_cls=False,
                        augment=False,
                        verbose=False,
                        model=None,
                        dataloader=None,
                        save_dir=Path(''),
                        save_txt=False,
                        save_hybrid=False,
                        save_conf=False,
                        plots=True,
                        wandb_logger=None,
                        compute_loss=None,
                        half_precision=True,
                        trace=False,
                        is_coco=False,
                        v5_metric=False,
                        # Integrated parameters
                        enable_categorization=True,
                        min_recall=0.8,
                        categorization_classes=None,
                        device='',
                        task='val',
                        project='runs/test',
                        name='exp',
                        exist_ok=False,
                        max_problem_images=5000,
                        debug_first_images=10,
                        enable_post_processing=True):
    """
    🎯 최종 통합 테스트 함수
    - 좌표 변환 문제 완전 해결
    - 메모리 효율적 2단계 처리
    - JSON 직렬화 안전
    - 상세한 매칭 정보 포함
    """
    
    logger = setup_logging()
    
    # Initialize/load model and set device
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        set_logging()
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        model = attempt_load(weights, map_location=device)
        gs = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=gs)
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half precision
    half = device.type != 'cpu' and half_precision
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    # 결과 수집 변수
    all_results = []
    category_stats = defaultdict(int)
    problem_count = 0
    
    # Create minimal opt object for dataloader
    class OptConfig:
        def __init__(self):
            self.rect = True
            self.cache_images = False
            self.image_weights = False
            self.quad = False
            self.prefix = ''
            self.workers = 8
            self.single_cls = single_cls
    
    opt_config = OptConfig()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        
        task = task if task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt_config, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    logger.info("🎯 Starting final integrated analysis...")

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1

            # 🎯 Integrated categorization analysis
            if enable_categorization:
                try:
                    # 디버깅 모드 설정
                    debug_mode = (seen <= debug_first_images)
                    
                    # 통합 평가 함수 호출
                    result = evaluate_detection_integrated(
                        pred if pred is not None and len(pred) > 0 else torch.empty((0, 6)),
                        labels,  # targets는 이미 pixel 좌표로 변환됨
                        path, names, conf_thres, iou_thres, 
                        (height, width), min_recall, categorization_classes, debug_mode
                    )
                    
                    # 통계 업데이트
                    category_stats[result.category] += 1
                    
                    # false/miss detection만 저장 (메모리 효율성)
                    if result.category in ['false_detect', 'miss_detect', 'low_conf', 'partial_detect', 'borderline_detect']:  # 🎯 카테고리 추가
                        all_results.append(result.to_dict())
                        problem_count += 1
                        
                        # 메모리 관리
                        if problem_count > max_problem_images:
                            keep_ratio = 0.8
                            keep_count = int(len(all_results) * keep_ratio)
                            all_results = random.sample(all_results, keep_count)
                            logger.warning(f"Memory management: kept {keep_count} results")
                    
                    # 주기적 로그
                    if seen % 1000 == 0:
                        logger.info(f"Processed {seen} images: Found {problem_count} problems")
                        current_stats = dict(category_stats)
                        logger.info(f"Current stats: {current_stats}")
                        
                except Exception as e:
                    if seen <= 10:  # 초기 몇 개만 로그
                        logger.warning(f"Error in categorization for {path}: {e}")

            # 원래 test.py 로직 계속... (기존과 동일)
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)

                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names.get(c, f'class_{c}'), seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print collection results
    if enable_categorization:
        total_images = sum(category_stats.values())
        logger.info(f"\n🎯 === FINAL INTEGRATED ANALYSIS COMPLETED ===")
        logger.info(f"📊 Total images processed: {total_images}")
        logger.info(f"🔍 Problem detections collected: {len(all_results)}")
        
        for category, count in category_stats.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            logger.info(f"📂 {category}: {count} ({percentage:.1f}%)")

    # 결과 저장 및 후처리
    if enable_categorization and all_results:
        logger.info(f"\n🎨 === STARTING INTEGRATED POST-PROCESSING ===")
        
        # 결과 저장
        results_file = save_dir / 'integrated_problem_detections.jsonl'
        logger.info(f"💾 Saving {len(all_results)} problem detections to {results_file}")
        
        with open(results_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result, default=str) + '\n')
        
        if enable_post_processing:
            # 후처리 실행
            try:
                post_process_integrated(
                    results_file=results_file,
                    save_dir=save_dir,
                    names=names,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    max_images_per_category=1000
                )
                logger.info("🎉 Integrated post-processing completed successfully!")
                
            except Exception as e:
                logger.error(f"❌ Error in post-processing: {e}")
                logger.info("⚠️  Basic test results are still available")

    # GPU 메모리 정리
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

    # Continue with plots and JSON saving
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = './coco/annotations/instances_val2017.json'
        pred_json = str(save_dir / f"{w}_predictions.json")
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Return results
    model.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    results = (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist())
    if enable_categorization:
        results = (*results, dict(category_stats))
    
    return results, maps, (t0, t1, t0 + t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_final_integrated.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.yaml path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    # Integrated parameters
    parser.add_argument('--enable-categorization', action='store_true', default=True, help='enable problem detection collection')
    parser.add_argument('--min-recall', type=float, default=0.8, help='minimum recall for good detection')
    parser.add_argument('--categorization-classes', nargs='+', type=int, help='classes to analyze for categorization')
    parser.add_argument('--max-problem-images', type=int, default=5000, help='maximum problem images to collect')
    parser.add_argument('--debug-first-images', type=int, default=10, help='number of first images to debug')
    parser.add_argument('--enable-post-processing', action='store_true', default=True, help='enable post-processing')
    
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)
    print(opt)

    try:
        results, maps, times = test_final_integrated(
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            opt.verbose,
            save_txt=opt.save_txt | opt.save_hybrid,
            save_hybrid=opt.save_hybrid,
            save_conf=opt.save_conf,
            trace=not opt.no_trace,
            v5_metric=opt.v5_metric,
            enable_categorization=opt.enable_categorization,
            min_recall=opt.min_recall,
            categorization_classes=opt.categorization_classes,
            max_problem_images=opt.max_problem_images,
            debug_first_images=opt.debug_first_images,
            enable_post_processing=opt.enable_post_processing,
            device=opt.device,
            task=opt.task,
            project=opt.project,
            name=opt.name,
            exist_ok=opt.exist_ok
        )
        
        print(f"\n🎉 Final integrated test completed successfully!")
        print(f"📊 Results: mP={results[0]:.3f}, mR={results[1]:.3f}, mAP50={results[2]:.3f}, mAP={results[3]:.3f}")
        if len(results) > 7:
            print(f"🎯 Integrated coordinate-fixed analysis completed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user (Ctrl+C)")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("✅ Process completed and GPU memory cleaned up")