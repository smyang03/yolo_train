import argparse
import time
import platform
from pathlib import Path
import logging
import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
import shutil
import random
from models.experimental import attempt_load
from utils.datasets import LoadImagestxt, read_label_file, create_result_dirs
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, 
    xyxy2xywh, xywhn2xyxy, strip_optimizer, 
    set_logging, increment_path, calculate_iou
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_model(weights, device, img_size, trace=True):
    """Load the detection model."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from {weights}")
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    
    # Enable half precision if on CUDA
    half = device.type != 'cpu'
    if half:
        model.half()
    
    model.eval()
    
    # Trace model for better performance if requested
    if trace and device.type != 'cpu':
        try:
            logger.info("Converting model to Traced-model...")
            # 입력과 모델의 타입을 일치시키기 위해 half precision으로 통일
            rand_example = torch.zeros(1, 3, img_size, img_size).to(device)
            if half:
                rand_example = rand_example.half()
            else:
                # 모델이 half가 아니라면 float32로 유지
                rand_example = rand_example.float()
                
            model = TracedModel(model, device, img_size, rand_example)
            logger.info("Successfully converted to Traced-model")
        except Exception as e:
            logger.warning(f"Failed to create traced model: {e}")
            logger.info("Using non-traced model instead")
    
    # Warmup
    if device.type != 'cpu':
        logger.info("Warming up model...")
        img = torch.zeros(1, 3, img_size, img_size).to(device)
        img = img.half() if half else img.float()
        _ = model(img)
    
    return model, stride, img_size, half


def process_batch(img, model, device, half, augment=False, conf_thres=0.25, iou_thres=0.45, 
                  classes=None, agnostic_nms=False):
    """Process a batch of images through the model."""
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        pred = model(img, augment=augment)[0]
    
    # Apply NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
    )
    
    return pred


def safe_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    안전하게 좌표를 스케일링하는 함수.
    원본 scale_coords 함수와 동일하지만 에러 처리가 강화됨.
    
    Args:
        img1_shape: 소스 이미지 형태 (높이, 너비)
        coords: 변환할 좌표 (xyxy 형식)
        img0_shape: 대상 이미지 형태 (높이, 너비)
        ratio_pad: 선택적 비율과 패딩 값
        
    Returns:
        변환된 좌표
    """
    # 입력 형태 확인 및 수정
    if isinstance(img0_shape, torch.Tensor):
        img0_shape = img0_shape.cpu().numpy()
    if isinstance(img1_shape, torch.Tensor):
        img1_shape = img1_shape.cpu().numpy()
        
    # 차원 확인
    if len(img0_shape) >= 3:  # 높이, 너비, 채널
        img0_h, img0_w = img0_shape[0], img0_shape[1]
    else:  # 높이, 너비
        img0_h, img0_w = img0_shape[0], img0_shape[1] if len(img0_shape) > 1 else img0_shape[0]
    
    if len(img1_shape) >= 3:
        img1_h, img1_w = img1_shape[0], img1_shape[1]
    else:
        img1_h, img1_w = img1_shape[0], img1_shape[1] if len(img1_shape) > 1 else img1_shape[0]
    
    # 좌표 복사하여 원본 유지
    coords_result = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
    
    # 스케일링 비율 계산
    if ratio_pad is None:
        gain = min(img1_h / img0_h, img1_w / img0_w)
        pad = (img1_w - img0_w * gain) / 2, (img1_h - img0_h * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    # 좌표 변환
    coords_result[:, [0, 2]] -= pad[0]  # x padding
    coords_result[:, [1, 3]] -= pad[1]  # y padding
    coords_result[:, :4] /= gain
    
    # 클리핑 (이미지 범위 내로 제한)
    coords_result[:, 0].clamp_(0, img0_w)  # x1
    coords_result[:, 1].clamp_(0, img0_h)  # y1
    coords_result[:, 2].clamp_(0, img0_w)  # x2
    coords_result[:, 3].clamp_(0, img0_h)  # y2
    
    return coords_result


def evaluate_detections(pred, gt_boxes, im0s, img, gn, names, conf_thres, iou_thres, colors=None, 
                        visualize=False, min_recall=0.8, classes=None):
    """
    Evaluate detections against ground truth, filtering for specific classes.
    
    Args:
        pred: Model predictions
        gt_boxes: Ground truth boxes
        im0s: Original image
        img: Preprocessed image tensor
        gn: Normalization gain
        names: Class names
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        colors: Colors for visualization
        visualize: Whether to generate visualization
        min_recall: Minimum recall for good detection
        classes: List of specific classes to evaluate. If None, all classes are evaluated.
        
    Returns:
        dict: Results info including detection category, precision, recall, etc.
    """
    logger = logging.getLogger(__name__)
    
    if colors is None:
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    result = {
        'category': 'unknown',
        'matched_gt': set(),
        'matched_pred': set(),
        'precision': 0,
        'recall': 0,
        'pred_info': [],
        'visualization': None
    }
    
    # Process first detection batch (should be single image)
    det = pred[0] if len(pred) > 0 and pred[0] is not None else None
    
    # Create debug visualization images
    im_pred = im0s.copy()  # Detection visualization
    im_gt = im0s.copy()    # Ground truth visualization
    
    # 클래스 필터링된 ground truth boxes만 선택
    filtered_gt_boxes = []
    classes = [0,1,2,3,4,5]
    if classes is not None:
        for gt_idx, gt_box in enumerate(gt_boxes):
            gt_cls = int(gt_box[0])
            if gt_cls in [int(cls) for cls in classes]:
                filtered_gt_boxes.append((gt_idx, gt_box))
    else:
        filtered_gt_boxes = [(gt_idx, gt_box) for gt_idx, gt_box in enumerate(gt_boxes)]
    
    # First add ground truth boxes to GT visualization
    for gt_idx, gt_box in filtered_gt_boxes:
        try:
            xyxy = xywhn2xyxy(torch.tensor(gt_box[1:]).unsqueeze(0),  # gn을 곱하지 않음
                                w=im_gt.shape[1], h=im_gt.shape[0]).view(-1).tolist()
            if visualize:
                cls_id = int(gt_box[0])
                label = f'{names[cls_id]} GT'
                plot_one_box(xyxy, im_gt, label=label, color=colors[cls_id], line_thickness=2)
        except Exception as e:
            logger.error(f"Error processing GT box {gt_idx}: {e}")
            continue
    
    # Match detections with ground truth
    all_conf_good = True  # 모든 검출 결과의 신뢰도가 좋은지 확인
    
    if det is not None and len(det) > 0:
        # 안전한 좌표 스케일링을 위한 수정
        try:
            # 이미지 형태 확인 및 스케일링 수행
            if len(im0s.shape) == 3:  # 색상 채널이 있는 이미지 (높이, 너비, 채널)
                img0_shape = im0s.shape[:2]  # (높이, 너비)만 사용
            else:
                img0_shape = im0s.shape  # 그레이스케일 이미지나 다른 형태
                
            if len(img.shape) == 4:  # 배치, 채널, 높이, 너비
                img1_shape = img.shape[2:]  # (높이, 너비)
            else:
                img1_shape = img.shape[1:]  # 첫 번째 차원이 없는 경우
                
            # 안전한 스케일링 함수 사용
            det[:, :4] = safe_scale_coords(img1_shape, det[:, :4], img0_shape).round()
        except Exception as e:
            logger.error(f"Error scaling coordinates: {e}")
            logger.debug(f"img shape: {img.shape}, im0s shape: {im0s.shape}")
            # 스케일링 실패 시 원본 좌표 유지
        
        # 클래스 필터링된 detection 결과만 선택
        filtered_det = []
        if classes is not None:
            for i, (*xyxy, conf, cls) in enumerate(det):
                cls_id = int(cls)
                if cls_id == 2 and cls_id == 5 and cls_id == 7:
                    cls_id = 1
                elif cls_id == 14:
                    cls_id = 2
                elif cls_id == 15 and cls_id == 16:
                    cls_id = 3
                    
                if cls_id in [int(cls) for cls in classes]:
                    filtered_det.append((i, (*xyxy, conf, cls)))
        else:
            filtered_det = [(i, d) for i, d in enumerate(det)]
        
        # 모든 검출 결과의 신뢰도 확인
        all_conf_good = all(conf >= conf_thres for _, (*_, conf, _) in filtered_det)
        
        # Add predictions to visualization image
        for pred_idx, (*xyxy, conf, cls) in filtered_det:
            cls_id = int(cls)
            pred_bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            result['pred_info'].append((cls_id, pred_bbox, conf.item()))
            
            if visualize:
                label = f'{names[cls_id]} {conf:.2f}'
                plot_one_box(xyxy, im_pred, label=label, color=colors[cls_id], line_thickness=2)
        
        # For each prediction, find the best matching GT box
        for pred_idx, (*xyxy, conf, cls) in filtered_det:
            pred_cls = int(cls)
            pred_bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            
            # Find best matching GT box
            best_iou = 0
            best_gt_idx = -1
            
            for orig_gt_idx, gt_box in filtered_gt_boxes:
                gt_cls = int(gt_box[0])
                
                # Only match if classes are the same
                if pred_cls == gt_cls:
                    iou = calculate_iou(pred_bbox, gt_box[1:])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = orig_gt_idx
            
            # If IoU is above threshold, consider it a match
            if best_iou > iou_thres and best_gt_idx >= 0:
                result['matched_gt'].add(best_gt_idx)
                result['matched_pred'].add(pred_idx)
    
    # Calculate precision and recall based on filtered ground truth and predictions
    filtered_gt_count = len(filtered_gt_boxes)
    filtered_pred_count = len(filtered_det) if det is not None and len(det) > 0 else 0
    
    if filtered_gt_count > 0:
        result['recall'] = len(result['matched_gt']) / filtered_gt_count
    else:
        result['recall'] = 1.0 if filtered_pred_count == 0 else 0.0
        
    if filtered_pred_count > 0:
        result['precision'] = len(result['matched_pred']) / filtered_pred_count
    else:
        result['precision'] = 1.0 if filtered_gt_count == 0 else 0.0
    
    # Determine result category
    if filtered_gt_count > 0:
        if result['recall'] >= min_recall:
            if all_conf_good:
                result['category'] = 'good_detect'
            else:
                result['category'] = 'low_conf'
        elif result['recall'] > 0 and result['recall'] < min_recall:
            result['category'] = 'miss_detect'
        else:
            result['category'] = 'false_detect'
    else:
        # No ground truth, but we have detections - false positive
        if filtered_pred_count > 0:
            result['category'] = 'false_detect'
        else:
            # No ground truth, no detections - correctly did nothing
            result['category'] = 'background'
    # Create combined visualization if requested
    if visualize:
        try:
            # Add headings to each image
            heading_height = 30
            h, w = im_pred.shape[:2]
            c = 3 if len(im_pred.shape) == 3 else 1
            
            # Create images with heading space
            im_pred_with_heading = np.zeros((h + heading_height, w, c), dtype=np.uint8)
            im_gt_with_heading = np.zeros((h + heading_height, w, c), dtype=np.uint8)
            
            # Add headings
            cv2.putText(im_pred_with_heading, f"Prediction (P:{result['precision']:.2f}, R:{result['recall']:.2f})", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(im_gt_with_heading, "Ground Truth", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Copy original images into the space beneath the headings
            im_pred_with_heading[heading_height:, :, :] = im_pred
            im_gt_with_heading[heading_height:, :, :] = im_gt
            
            # Combine side by side
            result['visualization'] = np.hstack((im_pred_with_heading, im_gt_with_heading))
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            result['visualization'] = None
    
    return result

def save_results(result, path, target_dir, label_path=None):
    """Save results to appropriate directories."""
    # Create paths
    p = Path(path)
    target_img_path = target_dir / 'JPEGImages' / p.name
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(target_img_path), exist_ok=True)
    
    # Copy image
    shutil.copy(str(path), str(target_img_path))
    
    # Copy label if it exists
    if label_path and os.path.exists(label_path):
        target_label_path = target_dir / 'labels' / (p.stem + '.txt')
        os.makedirs(os.path.dirname(target_label_path), exist_ok=True)
        shutil.copy(label_path, str(target_label_path))
    
    return target_img_path


def detect(opt):
    """Main detection function."""
    logger = setup_logging()
    logger.info(f"Starting detection with options: {opt}")
    is_windows = platform.system() == 'Windows'
    if opt.view_img and not is_windows:
        logger.warning("view_img option is only supported on Windows. Disabling.")

    # opt.view_img = True
    # opt.save_debug_images = True
    # Ensure CUDA is available if specified
    if opt.device.lower() != 'cpu' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        opt.device = 'cpu'
    
    # Set up save directory
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to file as well
    file_handler = logging.FileHandler(save_dir / 'detection_log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Initialize device
    device = select_device(opt.device)
    
    # Load model
    model, stride, imgsz, half = load_model(
        opt.weights, device, opt.img_size, trace=not opt.no_trace
    )
    
    # Get model class names and set up colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Set up dataset
    try:
        dataset = LoadImagestxt(opt.source, img_size=imgsz, stride=stride)
        logger.info(f"Dataset loaded with {len(dataset)} images")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create result directories
    try:
        result_dirs = create_result_dirs(opt.savedirs)
        debug_dirs = {}
        for category in ['good_detect', 'miss_detect', 'false_detect', 'background', 'low_conf']:
            debug_dir = result_dirs[category] / 'debug_images'
            debug_dir.mkdir(exist_ok=True, parents=True)
            debug_dirs[category] = debug_dir
            
        logger.info(f"Result directories created at {opt.savedirs}")
    except Exception as e:
        logger.error(f"Failed to create result directories: {e}")
        return
    
    # Statistics
    stats = {cat: 0 for cat in ['good_detect', 'miss_detect', 'false_detect','background','low_conf']}
    total_images = 0
    successful_images = 0
    t0 = time.time()
    
    # Process images
    for idx, (path, img, im0s, vid_cap) in enumerate(tqdm(dataset, desc="Processing images")):
        try:
            logger.info(f"Processing image {idx+1}/{len(dataset)}:Results: {stats}")
            total_images += 1
            
            # Load ground truth
            label_path = str(Path(path)).replace('JPEGImages', 'labels').replace('.jpg', '.txt')
            if not os.path.exists(label_path):
                # 다른 확장자도 시도
                label_path = label_path.replace('.txt', '.txt')
                if not os.path.exists(label_path):
                    logger.warning(f"Label file not found for {path}")
                    gt_boxes = []  # 빈 GT 박스 목록
                else:
                    gt_boxes = read_label_file(label_path)
            else:
                gt_boxes = read_label_file(label_path)
            
            # Process batch
            pred = process_batch(
                img, model, device, half,
                augment=opt.augment,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                classes=opt.classes,
                agnostic_nms=opt.agnostic_nms
            )
            
            # Get normalization gain for image
            if len(im0s.shape) == 3:  # (높이, 너비, 채널)
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # [w, h, w, h]
            else:  # (높이, 너비)
                h, w = im0s.shape
                gn = torch.tensor([w, h, w, h])
            
            # Evaluate detections
            result = evaluate_detections(
                pred, gt_boxes, im0s, img, gn, names, 
                opt.conf_thres, opt.iou_thres, colors,
                visualize=opt.view_img or opt.save_debug_images,
                min_recall=opt.min_recall,
                classes=opt.classes
            )
            
            # Update statistics
            stats[result['category']] += 1

            # Save results if needed
            if not opt.nosave:
                target_dir = result_dirs[result['category']]
                save_results(result, path, target_dir, label_path)
            
            # Save visualization if requested
            if opt.save_debug_images and result['visualization'] is not None:
                if random.random() < 0.01:
                   # 결과 카테고리에 따른 디버그 디렉토리에 저장
                   category = result['category']
                   debug_img_path = debug_dirs[category] / f"{Path(path).stem}_debug.jpg"
                   try:
                       cv2.imwrite(str(debug_img_path), result['visualization'])
                       logger.info(f"Saved debug image: {debug_img_path}")
                   except Exception as e:
                       logger.error(f"Failed to save debug image: {e}")
            
            # Show visualization if requested
            if opt.view_img and result['visualization'] is not None and is_windows:
                cv2.imshow('Detection Evaluation', cv2.resize(result['visualization'], (1600, 900)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            successful_images += 1
                
        except Exception as e:
            logger.error(f"Error processing image {path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Print final statistics
    processing_time = time.time() - t0
    logger.info(f"Processed {total_images} images in {processing_time:.2f}s ({processing_time/total_images:.3f}s per image)")
    logger.info(f"Successfully processed: {successful_images}/{total_images} images")
    logger.info(f"Results: {stats}")
    
    # Save summary
    with open(save_dir / 'summary.txt', 'w') as f:
        f.write(f"Total images: {total_images}\n")
        f.write(f"Successfully processed: {successful_images}\n")
        f.write(f"Processing time: {processing_time:.2f}s ({processing_time/total_images:.3f}s per image)\n")
        f.write(f"Good detections: {stats['good_detect']} ({stats['good_detect']/total_images:.1%})\n")
        f.write(f"Missed detections: {stats['miss_detect']} ({stats['miss_detect']/total_images:.1%})\n")
        f.write(f"False detections: {stats['false_detect']} ({stats['false_detect']/total_images:.1%})\n")
        f.write(f"low_conf detection: {stats['low_conf']} ({stats['low_conf']/total_images:.1%})\n")
        f.write(f"background: {stats['background']} ({stats['background']/total_images:.1%})\n")
    
    logger.info(f"Done. Results saved to {save_dir}")
    
    # Close windows
    if opt.view_img and is_windows:
        cv2.destroyAllWindows()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='D:\\code\\SIAV2-AI-MODEL\\models\\detectnetwork\\00.안전환경\\pt\SIAV2_Detector_YOLOV7_SafeEnv_V4.0.0_FP32_240903_BASE.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='Z:\\101.etc\\core\\core\\data1\\valid.txt', help='source image list txt file')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--min-recall', type=float, default=0.8, help='Minimum recall for good detection')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-debug-images', action='store_true', help='save debug images')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true',help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 3, 4],help='filter by class')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--savedirs', type=str, default='Z:\\101.etc\\core\\core\\data1', help='base directory for saving results')
    
    return parser.parse_args()


if __name__ == '__main__':
    try:
        opt = parse_args()
        print(opt)
        
        with torch.no_grad():
            detect(opt)
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()