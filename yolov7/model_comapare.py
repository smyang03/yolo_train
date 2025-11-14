import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import yaml
import json
import os
import time
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import defaultdict

# YOLO 관련 모듈 임포트
# 실제 실행 환경에 맞게 조정 필요
try:
    from models.yolo import Model
    from utils.general import non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, box_iou
    from utils.datasets import create_dataloader
    from utils.torch_utils import select_device
    from utils.metrics import ap_per_class
except ImportError:
    print("YOLO 모듈을 찾을 수 없습니다. 경로를 확인하세요.")
    print("이 스크립트는 YOLOv7 코드베이스 내에서 실행해야 합니다.")


class ModelComparer:
    def __init__(self, scratch_model_path, finetuned_model_path, data_yaml, img_size=640, 
                 conf_thres=0.25, iou_thres=0.45, batch_size=16, device='', opt=None):
        
        self.scratch_model_path = scratch_model_path
        self.finetuned_model_path = finetuned_model_path
        self.data_yaml = data_yaml
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.batch_size = batch_size
        self.opt=opt
        
        # 결과 저장 폴더 생성
        self.output_dir = Path('model_comparison_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # 디바이스 설정
        self.device = select_device(device)
        
        # 데이터 설정 로드
        with open(data_yaml, encoding='UTF8') as f:
            self.data_dict = yaml.safe_load(f)
        
        # 클래스 이름 로드
        self.class_names = self.data_dict['names']
        self.nc = len(self.class_names)
        
        print(f"모델 비교 분석을 시작합니다.")
        print(f"스크래치 모델: {scratch_model_path}")
        print(f"파인튜닝 모델: {finetuned_model_path}")
        print(f"클래스 개수: {self.nc}")
        print(f"클래스 이름: {self.class_names}")
        
        # 모델 로드
        self.load_models()

    def load_models(self):
        """두 모델을 로드합니다."""
        print("모델 로드 중...")
        
        # 스크래치 모델 로드
        try:
            ckpt = torch.load(self.scratch_model_path, map_location=self.device)
            self.scratch_model = ckpt['model'] if 'model' in ckpt else ckpt
            if hasattr(self.scratch_model, 'float'):
                self.scratch_model = self.scratch_model.float()
            self.scratch_model.to(self.device).eval()
        except Exception as e:
            print(f"스크래치 모델 로드 실패: {e}")
            raise
        
        # 파인튜닝 모델 로드
        try:
            ckpt = torch.load(self.finetuned_model_path, map_location=self.device)
            self.finetuned_model = ckpt['model'] if 'model' in ckpt else ckpt
            if hasattr(self.finetuned_model, 'float'):
                self.finetuned_model = self.finetuned_model.float()
            self.finetuned_model.to(self.device).eval()
        except Exception as e:
            print(f"파인튜닝 모델 로드 실패: {e}")
            raise
        
        print("모델 로드 완료!")
    def prepare_input(self, tensor):
        return tensor.to(self.device).float()
    def create_dataloader(self):
        """테스트 데이터 로더를 생성합니다."""
        try:
            with open(self.opt.hyp) as f:
                hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
            val_path = self.data_dict['val']
            gs = max(int(self.scratch_model.stride.max() if hasattr(self.scratch_model, 'stride') else 32), 32)
            # dataloader = create_dataloader(val_path, self.img_size, self.batch_size, gs, self.opt,
            #                               {'rect': True}, hyp=hyp, augment=False,
            #                               cache=False, pad=0.5, workers=8)[0]
            dataloader = create_dataloader(val_path, self.img_size, self.batch_size, gs,
                                        self.opt, augment=False, cache=False, 
                                        pad=0.5, workers=8)[0]
            return dataloader
        except Exception as e:
            print(f"데이터로더 생성 실패: {e}")
            raise
    def compare_performance(self, alpha=0.5):
        """
        WiSE-FT 방법으로 두 모델의 성능을 비교합니다.
        
        Args:
            alpha (float): 가중치 보간 계수 (0과 1 사이의 값)
        """
        print("WiSE-FT 방법으로 성능 비교 분석 중...")
        
        # 두 모델의 가중치 상태 사전 로드
        scratch_state_dict = self.scratch_model.state_dict()
        finetuned_state_dict = self.finetuned_model.state_dict()
        
        # WiSE-FT: 가중치 선형 보간
        wise_ft_state_dict = {}
        for key in scratch_state_dict.keys():
            if key in finetuned_state_dict:
                # 가중치 선형 보간
                wise_ft_state_dict[key] = (1 - alpha) * scratch_state_dict[key] + alpha * finetuned_state_dict[key]
            else:
                # 키가 없으면 원래 스크래치 모델의 가중치 사용
                wise_ft_state_dict[key] = scratch_state_dict[key]
        
        # WiSE-FT 모델 생성 및 가중치 로드
        wise_ft_model = copy.deepcopy(self.scratch_model)
        wise_ft_model.load_state_dict(wise_ft_state_dict)
        wise_ft_model.to(self.device).eval()
        
        # 데이터로더 생성
        dataloader = self.create_dataloader()
        
        # 성능 측정 변수 초기화
        stats_scratch = []
        stats_finetuned = []
        stats_wise_ft = []
        
        with torch.no_grad():
            for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc="성능 평가 중")):
                imgs = imgs.to(self.device).float()
                targets = targets.to(self.device)
                
                # 세 모델 추론 (스크래치, 파인튜닝, WiSE-FT)
                pred_scratch = self.scratch_model(imgs)
                pred_finetuned = self.finetuned_model(imgs)
                pred_wise_ft = wise_ft_model(imgs)
                
                # 튜플 처리
                if isinstance(pred_scratch, tuple):
                    pred_scratch = pred_scratch[0]
                if isinstance(pred_finetuned, tuple):
                    pred_finetuned = pred_finetuned[0]
                if isinstance(pred_wise_ft, tuple):
                    pred_wise_ft = pred_wise_ft[0]
                
                # NMS 적용
                pred_scratch = non_max_suppression(pred_scratch, self.conf_thres, self.iou_thres)
                pred_finetuned = non_max_suppression(pred_finetuned, self.conf_thres, self.iou_thres)
                pred_wise_ft = non_max_suppression(pred_wise_ft, self.conf_thres, self.iou_thres)
                
                # 결과 수집 로직 (기존 코드와 유사)
                ...
        
        # 성능 측정 및 시각화
        class_ap_scratch = self._calculate_ap(stats_scratch)
        class_ap_finetuned = self._calculate_ap(stats_finetuned)
        class_ap_wise_ft = self._calculate_ap(stats_wise_ft)
        
        # 결과 시각화
        plt.figure(figsize=(10, 6))
        x = np.arange(len(class_ap_scratch))
        width = 0.25
        
        plt.bar(x - width, class_ap_scratch, width, label='Scratch', color='blue')
        plt.bar(x, class_ap_finetuned, width, label='Fine-tuned', color='red')
        plt.bar(x + width, class_ap_wise_ft, width, label='WiSE-FT', color='green')
        
        plt.xlabel('Class')
        plt.ylabel('Average Precision')
        plt.title('Performance Comparison: Scratch vs Fine-tuned vs WiSE-FT')
        plt.xticks(x, [self.class_names[i] for i in range(len(class_ap_scratch))], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'wise_ft_performance_comparison.png')
        
        return class_ap_scratch, class_ap_finetuned, class_ap_wise_ft
    # def compare_performance(self):
    #     """두 모델의 성능을 비교하고 시각화합니다."""
    #     print("성능 비교 분석 중...")
        
    #     # 데이터로더 생성
    #     try:
    #         dataloader = self.create_dataloader()
    #     except Exception as e:
    #         print(f"데이터로더 생성 실패로 성능 비교를 건너뜁니다: {e}")
    #         return None, None
        
    #     # 성능 측정 변수 초기화
    #     stats_scratch = []
    #     stats_finetuned = []
        
    #     # 예측 및 라벨 수집
    #     all_preds_scratch = []
    #     all_preds_finetuned = []
    #     all_targets = []
        
    #     # 모델 추론 및 데이터 수집
    #     with torch.no_grad():
    #         for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc="성능 평가 중")):
    #             # 데이터 전처리
    #             imgs = imgs.to(self.device).float()
    #             targets = targets.to(self.device)
                
    #             # 모델 추론
    #             pred_scratch = self.scratch_model(imgs)
    #             pred_finetuned = self.finetuned_model(imgs)
                
    #             # 튜플 처리
    #             if isinstance(pred_scratch, tuple):
    #                 pred_scratch = pred_scratch[0]
    #             if isinstance(pred_finetuned, tuple):
    #                 pred_finetuned = pred_finetuned[0]
                
    #             # NMS 적용
    #             pred_scratch = non_max_suppression(pred_scratch, self.conf_thres, self.iou_thres)
    #             pred_finetuned = non_max_suppression(pred_finetuned, self.conf_thres, self.iou_thres)
                
    #             # 결과 수집
    #             for si, ((pred_s, pred_f), target) in enumerate(zip(zip(pred_scratch, pred_finetuned), targets)):
    #                 # 타겟 차원 확인 및 처리
    #                 if target.dim() == 1:
    #                     if target.numel() > 0:
    #                         labels = target.view(1, -1)
    #                         tcls = [int(target[1].item())] if len(target) > 1 else []
    #                     else:
    #                         continue
    #                 else:
    #                     labels = target[target[:, 0] > -1]
    #                     tcls = labels[:, 1].tolist() if len(labels) else []
                    
    #                 # 결과 수집
    #                 if len(tcls):
    #                     all_targets.append(labels)
                        
    #                 if pred_s is not None and len(pred_s):
    #                     all_preds_scratch.append((pred_s.clone(), torch.tensor(tcls).to(self.device)))
                    
    #                 if pred_f is not None and len(pred_f):
    #                     all_preds_finetuned.append((pred_f.clone(), torch.tensor(tcls).to(self.device)))
        
    #     # 테스트 데이터에 존재하는 클래스만 추출
    #     present_classes = set()
    #     for labels in all_targets:
    #         if labels.dim() > 1:  # 2차원 텐서인 경우
    #             present_classes.update(labels[:, 1].int().tolist())
    #         else:  # 1차원 텐서인 경우
    #             if len(labels) > 1:  # 클래스 인덱스가 있는지 확인
    #                 present_classes.add(int(labels[1].item()))
        
    #     present_classes = sorted(list(present_classes))
    #     print(f"테스트 데이터에 존재하는 클래스: {present_classes}")
        
    #     if not present_classes:
    #         print("테스트 데이터에 유효한 클래스가 없습니다. 성능 비교를 건너뜁니다.")
    #         return None, None
        
    #     # AP 계산을 위한 변수 초기화
    #     class_ap_scratch = np.zeros(self.nc)
    #     class_ap_finetuned = np.zeros(self.nc)
        
    #     # 각 클래스별 AP 계산 (존재하는 클래스만)
    #     for cls_idx in present_classes:
    #         if cls_idx >= self.nc:
    #             continue
                
    #         # 스크래치 모델 AP 계산
    #         tp_scratch, fp_scratch = 0, 0
    #         for pred, target_cls in all_preds_scratch:
    #             for *_, conf, pred_cls in pred:
    #                 if int(pred_cls.item()) == cls_idx:
    #                     if cls_idx in target_cls:
    #                         tp_scratch += 1
    #                     else:
    #                         fp_scratch += 1
            
    #         precision_scratch = tp_scratch / (tp_scratch + fp_scratch) if (tp_scratch + fp_scratch) > 0 else 0
    #         class_ap_scratch[cls_idx] = precision_scratch
            
    #         # 파인튜닝 모델 AP 계산
    #         tp_finetuned, fp_finetuned = 0, 0
    #         for pred, target_cls in all_preds_finetuned:
    #             for *_, conf, pred_cls in pred:
    #                 if int(pred_cls.item()) == cls_idx:
    #                     if cls_idx in target_cls:
    #                         tp_finetuned += 1
    #                     else:
    #                         fp_finetuned += 1
            
    #         precision_finetuned = tp_finetuned / (tp_finetuned + fp_finetuned) if (tp_finetuned + fp_finetuned) > 0 else 0
    #         class_ap_finetuned[cls_idx] = precision_finetuned
        
    #     # 결과 시각화 (존재하는 클래스만)
    #     # (시각화 코드는 기존과 유사하나 present_classes만 사용)
        
    #     return class_ap_scratch, class_ap_finetuned
    def compare_weight_distributions(self):
        """두 모델의 가중치 분포를 비교하고 시각화합니다."""
        print("가중치 분포 비교 분석 중...")
        
        try:
            # 모델 가중치 추출
            scratch_weights = self.scratch_model.state_dict()
            finetuned_weights = self.finetuned_model.state_dict()
            
            # 공통 레이어 찾기
            common_layers = set(scratch_weights.keys()) & set(finetuned_weights.keys())
            
            # 가중치 차이 계산
            weight_diff = {}
            for layer in common_layers:
                if scratch_weights[layer].shape == finetuned_weights[layer].shape:
                    # 절대 차이의 평균 계산
                    diff = torch.abs(scratch_weights[layer].float() - finetuned_weights[layer].float())
                    weight_diff[layer] = diff.mean().item()
            
            # 차이가 큰 순서로 정렬
            sorted_diff = sorted(weight_diff.items(), key=lambda x: x[1], reverse=True)
            
            # 시각화
            plt.figure(figsize=(14, 10))
            
            # 1. 상위 20개 레이어 가중치 차이
            plt.subplot(2, 1, 1)
            top_layers = sorted_diff[:20]
            
            # 레이어 이름 준비
            layer_names = [x[0] for x in top_layers]
            layer_names = [name[-25:] if len(name) > 25 else name for name in layer_names]  # 이름 길이 제한
            
            # 출력 레이어(클래스 관련) 강조
            colors = ['skyblue' for _ in range(len(top_layers))]
            
            for i, layer_name in enumerate(layer_names):
                # 출력 레이어나 클래스 관련 레이어 강조 (예: "24.m.0" 등)
                if any(class_keyword in layer_name for class_keyword in ['24.m', '23.m', 'cls', 'class']):
                    colors[i] = 'orange'
            
            # 막대 그래프
            plt.barh(range(len(top_layers)), [x[1] for x in top_layers], color=colors)
            plt.yticks(range(len(top_layers)), layer_names)
            plt.xlabel('Mean Absolute Weight Difference')
            plt.title('Top 20 Layers with Largest Weight Differences')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # 2. 레이어 유형별 평균 가중치 차이
            plt.subplot(2, 1, 2)
            
            # 레이어 유형 분류
            layer_types = {}
            for layer, diff in weight_diff.items():
                # 레이어 유형 분류 로직 (간단한 예시)
                if 'conv' in layer:
                    layer_type = 'Convolution'
                elif 'bn' in layer:
                    layer_type = 'BatchNorm'
                elif 'm.0' in layer or 'm.1' in layer or 'm.2' in layer:
                    layer_type = 'Detection Head'
                elif 'bias' in layer:
                    layer_type = 'Bias'
                else:
                    layer_type = 'Other'
                
                if layer_type not in layer_types:
                    layer_types[layer_type] = []
                layer_types[layer_type].append(diff)
            
            # 레이어 유형별 평균 계산
            layer_type_avg = {k: np.mean(v) for k, v in layer_types.items()}
            
            # 막대 그래프
            plt.bar(layer_type_avg.keys(), layer_type_avg.values())
            plt.xlabel('layer type')
            plt.ylabel('average layer weight difference')
            plt.title('layer type average layer weight difference')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'weight_distribution_comparison.png', dpi=200)
            
            print(f"가중치 분포 비교 분석 완료. 결과 저장됨: {self.output_dir / 'weight_distribution_comparison.png'}")
            
            return sorted_diff
            
        except Exception as e:
            print(f"가중치 분포 비교 실패: {e}")
            return None

    def compare_feature_maps(self, test_image_path):
        """두 모델의 피처맵을 비교하고 시각화합니다."""
        print(f"피처맵 비교 분석 중... 이미지: {test_image_path}")
        
        if not os.path.exists(test_image_path):
            print(f"테스트 이미지를 찾을 수 없습니다: {test_image_path}")
            return
        
        try:
            # 이미지 로드 및 전처리
            img = cv2.imread(test_image_path)
            if img is None:
                print(f"이미지를 로드할 수 없습니다: {test_image_path}")
                return
                
            img_orig = img.copy()
            
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_for_vis = img.copy()  # 시각화용 이미지 저장
            
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if len(img.shape) == 3:
                img = img.unsqueeze(0)  # 배치 차원 추가
            
            # 모델 구조 출력 (디버깅용)
            print("모델 구조 탐색 중...")
            
            # 특징맵 추출을 위한 후크 등록
            feature_maps_scratch = {}
            feature_maps_finetuned = {}
            
            hooks_scratch = []
            hooks_finetuned = []
            
            # 모니터링할 레이어 자동 탐색
            monitor_layers = self._discover_monitor_layers()
            
            if not monitor_layers:
                print("모니터링할 레이어를 찾을 수 없습니다. 기본 시각화만 생성합니다.")
            else:
                print(f"모니터링할 레이어: {list(monitor_layers.keys())}")
                
                # 후크 함수 정의
                def get_features_scratch(name):
                    def hook(module, input, output):
                        feature_maps_scratch[name] = output.detach()
                    return hook
                
                def get_features_finetuned(name):
                    def hook(module, input, output):
                        feature_maps_finetuned[name] = output.detach()
                    return hook
                
                # 후크 등록
                for name, module_path in monitor_layers.items():
                    try:
                        if module_path is not None:
                            module_scratch = self._get_module_by_path(self.scratch_model, module_path)
                            module_finetuned = self._get_module_by_path(self.finetuned_model, module_path)
                            
                            if module_scratch is not None and module_finetuned is not None:
                                hooks_scratch.append(module_scratch.register_forward_hook(get_features_scratch(name)))
                                hooks_finetuned.append(module_finetuned.register_forward_hook(get_features_finetuned(name)))
                                print(f"레이어 '{name}' 후크 등록 성공")
                            else:
                                print(f"레이어 '{name}' 접근 실패: 모듈이 None입니다")
                    except Exception as e:
                        print(f"레이어 '{name}' 후크 등록 실패: {e}")
            
            # 모델 추론
            with torch.no_grad():
                # 예측 수행
                pred_scratch = self.scratch_model(img)
                pred_finetuned = self.finetuned_model(img)

                # 출력이 튜플인 경우 처리
                if isinstance(pred_scratch, tuple):
                    pred_scratch = pred_scratch[0]
                if isinstance(pred_finetuned, tuple):
                    pred_finetuned = pred_finetuned[0]
                    
                # NMS 적용
                pred_scratch = non_max_suppression(pred_scratch, self.conf_thres, self.iou_thres)
                pred_finetuned = non_max_suppression(pred_finetuned, self.conf_thres, self.iou_thres)
            
            # 후크 제거
            for hook in hooks_scratch:
                hook.remove()
            for hook in hooks_finetuned:
                hook.remove()
            
            # 피처맵 시각화
            plt.figure(figsize=(20, 15))
            
            # 원본 이미지 시각화
            plt.subplot(4, max(1, len(monitor_layers)), 1)
            plt.imshow(cv2.cvtColor(img_for_vis, cv2.COLOR_BGR2RGB))
            plt.title("Origin")
            plt.axis('off')
            
            # 스크래치 모델 예측 시각화
            plt.subplot(4, max(1, len(monitor_layers)), 2)
            plt_img = img_for_vis.copy()
            
            if pred_scratch[0] is not None and len(pred_scratch[0]):
                for *xyxy, conf, cls in pred_scratch[0]:
                    # 예측 박스 그리기
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(plt_img, c1, c2, (0, 255, 0), 2)
                    
                    # 클래스 라벨 및 신뢰도 표시
                    cls_idx = int(cls)
                    if cls_idx < len(self.class_names):
                        label = f'{self.class_names[cls_idx]} {conf:.2f}'
                    else:
                        label = f'Class {cls_idx} {conf:.2f}'
                    
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                    cv2.rectangle(plt_img, c1, c2, (0, 255, 0), -1)
                    cv2.putText(plt_img, label, (c1[0], c1[1] + t_size[1] + 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], 1)
            
            plt.imshow(cv2.cvtColor(plt_img, cv2.COLOR_BGR2RGB))
            plt.title("Scratch Model Inference")
            plt.axis('off')
            
            # 파인튜닝 모델 예측 시각화
            plt.subplot(4, max(1, len(monitor_layers)), 3)
            plt_img = img_for_vis.copy()
            
            if pred_finetuned[0] is not None and len(pred_finetuned[0]):
                for *xyxy, conf, cls in pred_finetuned[0]:
                    # 예측 박스 그리기
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(plt_img, c1, c2, (255, 0, 0), 2)
                    
                    # 클래스 라벨 및 신뢰도 표시
                    cls_idx = int(cls)
                    if cls_idx < len(self.class_names):
                        label = f'{self.class_names[cls_idx]} {conf:.2f}'
                    else:
                        label = f'Class {cls_idx} {conf:.2f}'
                    
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                    cv2.rectangle(plt_img, c1, c2, (255, 0, 0), -1)
                    cv2.putText(plt_img, label, (c1[0], c1[1] + t_size[1] + 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], 1)
            
            plt.imshow(cv2.cvtColor(plt_img, cv2.COLOR_BGR2RGB))
            plt.title("Fine-tuned Model Inference")
            plt.axis('off')
            
            # 피처맵이 존재하는 경우 시각화
            if monitor_layers and (feature_maps_scratch or feature_maps_finetuned):
                # 레이어별 피처맵 시각화
                for i, layer_name in enumerate(monitor_layers.keys()):
                    if layer_name in feature_maps_scratch and layer_name in feature_maps_finetuned:
                        # 스크래치 모델 피처맵
                        plt.subplot(4, len(monitor_layers), i+len(monitor_layers)+1)
                        
                        # 피처맵 처리 (채널 평균)
                        feat_scratch = feature_maps_scratch[layer_name][0].cpu().numpy()
                        feat_mean = np.mean(feat_scratch, axis=0)
                        
                        # 정규화
                        feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-10)
                        
                        plt.imshow(feat_mean, cmap='viridis')
                        plt.title(f'Scratch: {layer_name}')
                        plt.axis('off')
                        
                        # 파인튜닝 모델 피처맵
                        plt.subplot(4, len(monitor_layers), i+2*len(monitor_layers)+1)
                        
                        # 피처맵 처리 (채널 평균)
                        feat_finetuned = feature_maps_finetuned[layer_name][0].cpu().numpy()
                        feat_mean = np.mean(feat_finetuned, axis=0)
                        
                        # 정규화
                        feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-10)
                        
                        plt.imshow(feat_mean, cmap='viridis')
                        plt.title(f'Fine-tuned: {layer_name}')
                        plt.axis('off')
                        
                        # 피처맵 차이
                        plt.subplot(4, len(monitor_layers), i+3*len(monitor_layers)+1)
                        
                        # 피처맵 차이 계산
                        feat_scratch = feature_maps_scratch[layer_name][0].cpu().numpy()
                        feat_finetuned = feature_maps_finetuned[layer_name][0].cpu().numpy()
                        
                        # 채널 수가 다를 경우를 대비한 처리
                        min_channels = min(feat_scratch.shape[0], feat_finetuned.shape[0])
                        feat_scratch = feat_scratch[:min_channels]
                        feat_finetuned = feat_finetuned[:min_channels]
                        
                        # 평균 계산
                        feat_scratch_mean = np.mean(feat_scratch, axis=0)
                        feat_finetuned_mean = np.mean(feat_finetuned, axis=0)
                        
                        # 차이 계산 및 정규화
                        diff = np.abs(feat_finetuned_mean - feat_scratch_mean)
                        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-10)
                        
                        plt.imshow(diff, cmap='hot')
                        plt.title(f'Difference: {layer_name}')
                        plt.axis('off')
            else:
                print("피처맵이 생성되지 않았습니다. 기본 시각화만 적용됩니다.")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_map_comparison.png', dpi=200)
            
            print(f"피처맵 비교 분석 완료. 결과 저장됨: {self.output_dir / 'feature_map_comparison.png'}")
            
        except Exception as e:
            print(f"피처맵 비교 실패: {e}")
            import traceback
            traceback.print_exc()

    def _discover_monitor_layers(self):
        """모델 구조를 탐색하여 모니터링할 레이어를 자동으로 찾습니다."""
        monitor_layers = {}
        
        # YOLOv7 구조 탐색 
        try:
            # 모델 타입 확인
            model_type = type(self.scratch_model).__name__
            print(f"모델 타입: {model_type}")
            
            # 1. YOLOv7 모델의 경우 - 모듈 경로 찾기
            if hasattr(self.scratch_model, 'model'):
                # 모델 내부 구조 탐색
                model_layers = self._explore_model_structure(self.scratch_model)
                
                # 모델 레이어 타입 카운트 (레이어 종류 파악)
                layer_type_counts = defaultdict(int)
                for path, module in model_layers.items():
                    layer_type = type(module).__name__
                    layer_type_counts[layer_type] += 1
                
                print(f"레이어 타입 분포: {dict(layer_type_counts)}")
                
                # 주요 레이어 타입 (Conv2d, BatchNorm2d 등) 필터링
                conv_layers = {path: module for path, module in model_layers.items() 
                            if isinstance(module, torch.nn.Conv2d)}
                
                # 레이어 깊이에 따라 분류
                if conv_layers:
                    total_layers = len(conv_layers)
                    layer_paths = sorted(conv_layers.keys())
                    
                    # 특정 위치의 레이어 선택 (예: 20%, 50%, 80% 위치)
                    indices = [
                        int(total_layers * 0.2),  # 초기 (20%)
                        int(total_layers * 0.5),  # 중간 (50%)
                        int(total_layers * 0.8),  # 후기 (80%)
                    ]
                    
                    # 중복 방지
                    indices = sorted(set(indices))
                    
                    # 선택된 레이어 경로 저장
                    for i, idx in enumerate(indices):
                        if 0 <= idx < len(layer_paths):
                            layer_name = f"layer_{i+1}"  # 레이어 이름 (layer_1, layer_2, ...)
                            monitor_layers[layer_name] = layer_paths[idx]
            
            # 모델 구조에 맞는 레이어가 없으면 일부 고정 위치 시도
            if not monitor_layers:
                print("기본 구조 탐색으로 레이어를 찾을 수 없어 고정 레이어 시도")
                common_paths = [
                    "model.0",                # 첫 번째 레이어
                    "model.model.0",          # 중첩된 모델 구조
                    "model.backbone.0",       # 백본 첫 레이어
                    "model.module.0",         # DP/DDP 모델 구조
                    "model.module.model.0",   # 중첩된 DP/DDP 모델 구조
                ]
                
                for i, path in enumerate(common_paths):
                    module = self._get_module_by_path(self.scratch_model, path)
                    if module is not None:
                        monitor_layers[f"fixed_layer_{i+1}"] = path
                
        except Exception as e:
            print(f"레이어 탐색 중 오류: {e}")
        
        return monitor_layers

    def _explore_model_structure(self, model, prefix="", max_depth=10):
        """모델 구조를 재귀적으로 탐색하여 모든 모듈의 경로를 반환합니다."""
        if max_depth <= 0:
            return {}
        
        modules_dict = {}
        
        # 기본 케이스: named_children이 없는 경우
        if not list(model.named_children()):
            return {prefix: model} if prefix else {}
        
        # 재귀 케이스: 하위 모듈이 있는 경우
        for name, module in model.named_children():
            current_prefix = f"{prefix}.{name}" if prefix else name
            
            # 현재 모듈 저장
            modules_dict[current_prefix] = module
            
            # 하위 모듈 탐색 (재귀)
            child_modules = self._explore_model_structure(module, current_prefix, max_depth - 1)
            modules_dict.update(child_modules)
        
        return modules_dict

    def _get_module_by_path(self, model, path):
        """경로 문자열(예: 'model.0.conv')을 통해 모델의 특정 모듈에 접근합니다."""
        try:
            parts = path.split('.')
            current = model
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif part.isdigit() and hasattr(current, '__getitem__'):
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return None
                else:
                    return None
            
            return current
        except Exception:
            return None
    def generate_confusion_matrices(self, dataloader=None):
        """두 모델의 혼동 행렬을 생성하고 비교합니다."""
        print("혼동 행렬 분석 중...")
        
        try:
            # 데이터로더 생성 (없는 경우)
            if dataloader is None:
                try:
                    dataloader = self.create_dataloader()
                except Exception as e:
                    print(f"데이터로더 생성 실패로 혼동 행렬 생성을 건너뜁니다: {e}")
                    return None, None
            
            # 혼동 행렬 초기화
            confusion_matrix_scratch = np.zeros((self.nc, self.nc))
            confusion_matrix_finetuned = np.zeros((self.nc, self.nc))
            
            # 예측 및 라벨 수집
            n_batches = min(100, len(dataloader))  # 처리할 최대 배치 수 (메모리 제한)
            
            with torch.no_grad():
                for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc="혼동 행렬 생성", total=n_batches)):
                    if batch_i >= n_batches:  # 최대 배치 수 제한
                        break
                        
                    imgs = imgs.to(self.device).float()
                    targets = targets.to(self.device)
                    
                    # 모델 추론
                    pred_scratch = self.scratch_model(imgs)
                    pred_finetuned = self.finetuned_model(imgs)
                    
                    # NMS 적용
                    if isinstance(pred_scratch, tuple):
                        pred_scratch = pred_scratch[0]  # 첫 번째 요소는 감지 결과여야 함
                    if isinstance(pred_finetuned, tuple):
                        pred_finetuned = pred_finetuned[0]
                    pred_scratch = non_max_suppression(pred_scratch, self.conf_thres, self.iou_thres)
                    pred_finetuned = non_max_suppression(pred_finetuned, self.conf_thres, self.iou_thres)
                    
                    # 배치 내 각 이미지 처리
                    for si, ((pred_s, pred_f), target) in enumerate(zip(zip(pred_scratch, pred_finetuned), targets)):
                        # 차원 확인 및 타겟 처리
                        if target.dim() == 1:
                            # 1차원 텐서 처리
                            if target.numel() > 0:
                                # 클래스 인덱스는 일반적으로 두 번째 요소(인덱스 1)에 있음
                                if len(target) > 1:
                                    tcls = [int(target[1].item())]
                                    # 클래스 인덱스가 유효한 범위인지 확인
                                    if tcls[0] < self.nc:
                                        # 스크래치 모델 예측 처리
                                        if pred_s is not None and len(pred_s):
                                            pcls = pred_s[:, 5].cpu().numpy().astype(np.int32)
                                            for tc in tcls:
                                                for pc in pcls:
                                                    if pc < self.nc:
                                                        confusion_matrix_scratch[tc, pc] += 1
                                        
                                        # 파인튜닝 모델 예측 처리
                                        if pred_f is not None and len(pred_f):
                                            pcls = pred_f[:, 5].cpu().numpy().astype(np.int32)
                                            for tc in tcls:
                                                for pc in pcls:
                                                    if pc < self.nc:
                                                        confusion_matrix_finetuned[tc, pc] += 1
                        else:
                            # 2차원 텐서 처리 (기존 코드)
                            try:
                                labels = target[target[:, 0] > -1].cpu().numpy()
                                tcls = labels[:, 1].astype(np.int32) if len(labels) else []
                                
                                # 스크래치 모델 예측 처리
                                if pred_s is not None and len(pred_s) and len(tcls):
                                    pcls = pred_s[:, 5].cpu().numpy().astype(np.int32)
                                    for tc in tcls:
                                        if tc < self.nc:  # 유효한 클래스 인덱스 확인
                                            for pc in pcls:
                                                if pc < self.nc:  # 유효한 클래스 인덱스 확인
                                                    confusion_matrix_scratch[tc, pc] += 1
                                
                                # 파인튜닝 모델 예측 처리
                                if pred_f is not None and len(pred_f) and len(tcls):
                                    pcls = pred_f[:, 5].cpu().numpy().astype(np.int32)
                                    for tc in tcls:
                                        if tc < self.nc:  # 유효한 클래스 인덱스 확인
                                            for pc in pcls:
                                                if pc < self.nc:  # 유효한 클래스 인덱스 확인
                                                    confusion_matrix_finetuned[tc, pc] += 1
                            except IndexError as e:
                                print(f"타겟 처리 중 인덱스 오류: {e}, 타겟 형태: {target.shape}, 타입: {target.dtype}")
                            except Exception as e:
                                print(f"타겟 처리 중 기타 오류: {e}")
                
            # 행별 정규화 (각 실제 클래스에 대한 분포)
            row_sums_scratch = confusion_matrix_scratch.sum(axis=1, keepdims=True)
            row_sums_finetuned = confusion_matrix_finetuned.sum(axis=1, keepdims=True)
            
            # 0으로 나누기 방지
            confusion_norm_scratch = np.zeros_like(confusion_matrix_scratch)
            confusion_norm_finetuned = np.zeros_like(confusion_matrix_finetuned)
            
            # 합이 0이 아닌 행만 정규화
            non_zero_rows_scratch = (row_sums_scratch != 0).flatten()
            non_zero_rows_finetuned = (row_sums_finetuned != 0).flatten()
            
            if np.any(non_zero_rows_scratch):
                confusion_norm_scratch[non_zero_rows_scratch] = (
                    confusion_matrix_scratch[non_zero_rows_scratch] / row_sums_scratch[non_zero_rows_scratch]
                )
            
            if np.any(non_zero_rows_finetuned):
                confusion_norm_finetuned[non_zero_rows_finetuned] = (
                    confusion_matrix_finetuned[non_zero_rows_finetuned] / row_sums_finetuned[non_zero_rows_finetuned]
                )
            
            # 혼동 행렬 시각화
            plt.figure(figsize=(16, 7))
            
            # 스크래치 모델 혼동 행렬
            plt.subplot(1, 2, 1)
            sns.heatmap(confusion_norm_scratch, annot=False, vmin=0, vmax=1, 
                        cmap='Blues', square=True, xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Scratch Model Confusion Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45)
            
            # 파인튜닝 모델 혼동 행렬
            plt.subplot(1, 2, 2)
            sns.heatmap(confusion_norm_finetuned, annot=False, vmin=0, vmax=1, 
                        cmap='Reds', square=True, xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('fine Model Confusion Matrix') 
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=200)
            
            # 특정 클래스(파인튜닝된 클래스)에 대한 더 자세한 분석
            weight_diff = self.compare_weight_distributions()
            fine_tuned_classes = self._detect_tuned_classes(weight_diff)
            
            plt.figure(figsize=(12, 5 * len(fine_tuned_classes)))
            
            for i, cls_idx in enumerate(fine_tuned_classes):
                if cls_idx >= self.nc:
                    continue
                    
                # 클래스명 준비
                class_name = self.class_names[cls_idx]
                
                # 해당 클래스 행 추출
                row_scratch = confusion_norm_scratch[cls_idx]
                row_finetuned = confusion_norm_finetuned[cls_idx]
                
                # 차이 계산
                diff = row_finetuned - row_scratch
                
                # 시각화
                plt.subplot(len(fine_tuned_classes), 1, i+1)
                
                x = np.arange(self.nc)
                width = 0.35
                
                plt.bar(x - width/2, row_scratch, width, label='scratch', color='blue', alpha=0.7)
                plt.bar(x + width/2, row_finetuned, width, label='fine', color='red', alpha=0.7)
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.xlabel('Predicted class')
                plt.ylabel('Normalized Prediction Rate')
                plt.title(f'Model Prediction Comparison for Class "{class_name}" (Index: {cls_idx})')
                plt.xticks(x, self.class_names, rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'finetuned_classes_prediction_comparison.png', dpi=200)
            
            print(f"혼동 행렬 분석 완료. 결과 저장됨: {self.output_dir / 'confusion_matrices.png'}")
            
            return confusion_matrix_scratch, confusion_matrix_finetuned
            
        except Exception as e:
            print(f"혼동 행렬 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    def analyze_class_activations(self, test_images):
        """두 모델의 클래스별 활성화 패턴을 분석합니다."""
        print("클래스별 활성화 패턴 분석 중...")
        
        if not test_images or len(test_images) == 0:
            print("테스트 이미지가 제공되지 않았습니다.")
            return
        
        # 존재하는 이미지만 필터링
        valid_images = [img for img in test_images if os.path.exists(img)]
        if not valid_images:
            print("유효한 테스트 이미지가 없습니다.")
            return
            
        try:
            # 클래스별 신뢰도 저장소
            class_confs_scratch = defaultdict(list)
            class_confs_finetuned = defaultdict(list)
            
            # 클래스별 검출 횟수
            class_counts_scratch = defaultdict(int)
            class_counts_finetuned = defaultdict(int)
            
            # 각 테스트 이미지에 대해 처리
            for img_path in tqdm(valid_images, desc="이미지 분석"):
                # 이미지 로드 및 전처리
                img = cv2.imread(img_path)
                if img is None:
                    print(f"이미지를 로드할 수 없습니다: {img_path}")
                    continue
                    
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device).float()
                img = img.float() / 255.0
                img = img.unsqueeze(0)  # 배치 차원 추가
                
                # 모델 추론
                with torch.no_grad():
                    # 모델 출력 및 튜플 처리
                    pred_scratch = self.scratch_model(img)
                    pred_finetuned = self.finetuned_model(img)
                    
                    # 출력이 튜플인 경우 처리
                    if isinstance(pred_scratch, tuple):
                        pred_scratch = pred_scratch[0]  # 첫 번째 요소(탐지 결과)만 사용
                    if isinstance(pred_finetuned, tuple):
                        pred_finetuned = pred_finetuned[0]  # 첫 번째 요소만 사용
                    
                    # NMS 적용
                    pred_scratch = non_max_suppression(pred_scratch, self.conf_thres, self.iou_thres)
                    pred_finetuned = non_max_suppression(pred_finetuned, self.conf_thres, self.iou_thres)
                
                # 스크래치 모델 예측 처리
                if pred_scratch[0] is not None and len(pred_scratch[0]):
                    for *box, conf, cls_idx in pred_scratch[0].cpu().numpy():
                        cls_idx = int(cls_idx)
                        if cls_idx < self.nc:  # 유효한 클래스 인덱스인지 확인
                            class_confs_scratch[cls_idx].append(float(conf))
                            class_counts_scratch[cls_idx] += 1
                
                # 파인튜닝 모델 예측 처리
                if pred_finetuned[0] is not None and len(pred_finetuned[0]):
                    for *box, conf, cls_idx in pred_finetuned[0].cpu().numpy():
                        cls_idx = int(cls_idx)
                        if cls_idx < self.nc:  # 유효한 클래스 인덱스인지 확인
                            class_confs_finetuned[cls_idx].append(float(conf))
                            class_counts_finetuned[cls_idx] += 1
            
            # 분석할 데이터가 있는지 확인
            if not class_confs_scratch and not class_confs_finetuned:
                print("분석할 클래스 데이터가 없습니다. 이미지에서 객체가 탐지되지 않았을 수 있습니다.")
                return None, None
            
            # 클래스별 신뢰도 분포 시각화
            plt.figure(figsize=(15, 10))
            
            # 파인튜닝한 클래스 (0번, 5번)
            fine_tuned_classes = self._detect_tuned_classes()

            
            # 모든 클래스 순회
            all_classes = sorted(set(list(class_confs_scratch.keys()) + list(class_confs_finetuned.keys())))
            
            # 분석할 클래스 제한 (데이터가 있는 클래스만)
            analysis_classes = [cls for cls in all_classes if 
                                (cls in class_confs_scratch and len(class_confs_scratch[cls]) > 0) or 
                                (cls in class_confs_finetuned and len(class_confs_finetuned[cls]) > 0)]
            
            if not analysis_classes:
                print("분석할 클래스가 없습니다.")
                return None, None
            
            # 최대 9개 클래스만 표시 (분석 클래스가 9개를 초과하는 경우)
            if len(analysis_classes) > 9:
                # 파인튜닝한 클래스를 우선적으로 포함
                priority_classes = [cls for cls in fine_tuned_classes if cls in analysis_classes]
                other_classes = [cls for cls in analysis_classes if cls not in fine_tuned_classes]
                # 파인튜닝 클래스 + 나머지 클래스 중 일부 (총 9개까지)
                analysis_classes = priority_classes + other_classes[:9-len(priority_classes)]
            
            # 서브플롯 그리드 크기 계산
            grid_size = 3  # 3x3 그리드
            num_plots = min(9, len(analysis_classes))
            
            # 클래스별 신뢰도 분포 그래프
            for i, cls_idx in enumerate(analysis_classes[:num_plots]):
                plt.subplot(grid_size, grid_size, i+1)
                
                # 클래스명 준비
                class_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Class {cls_idx}"
                
                # 스크래치 모델 신뢰도
                if cls_idx in class_confs_scratch and len(class_confs_scratch[cls_idx]) > 0:
                    plt.hist(class_confs_scratch[cls_idx], bins=20, alpha=0.5, label='스크래치', color='blue')
                
                # 파인튜닝 모델 신뢰도
                if cls_idx in class_confs_finetuned and len(class_confs_finetuned[cls_idx]) > 0:
                    plt.hist(class_confs_finetuned[cls_idx], bins=20, alpha=0.5, label='파인튜닝', color='red')
                
                # 파인튜닝한 클래스 표시
                title = class_name
                if cls_idx in fine_tuned_classes:
                    title += " (fine)"
                    plt.axvline(x=0.5, color='green', linestyle='--', alpha=0.7)
                
                plt.title(title)
                plt.xlabel('Confidence')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'class_confidence_distribution.png', dpi=200)
            
            # 클래스별 검출 횟수 비교
            plt.figure(figsize=(12, 6))
            
            # 데이터 준비
            all_classes = sorted(set(list(class_counts_scratch.keys()) + list(class_counts_finetuned.keys())))
            class_names = [self.class_names[i] if i < len(self.class_names) else f"Class {i}" for i in all_classes]
            
            # 검출 횟수
            scratch_counts = [class_counts_scratch[cls] for cls in all_classes]
            finetuned_counts = [class_counts_finetuned[cls] for cls in all_classes]
            
            # 막대 그래프 데이터 준비
            x = np.arange(len(all_classes))
            width = 0.35
            
            plt.bar(x - width/2, scratch_counts, width, label='scratch', color='blue', alpha=0.7)
            plt.bar(x + width/2, finetuned_counts, width, label='fine', color='red', alpha=0.7)
            
            # 파인튜닝한 클래스 강조
            for i, cls_idx in enumerate(all_classes):
                if cls_idx in fine_tuned_classes:
                    plt.text(i, max(scratch_counts[i], finetuned_counts[i]) + 1, '★', 
                            ha='center', va='bottom', color='green', fontsize=15)
            
            plt.xlabel('class')
            plt.ylabel('Detection Count')
            plt.title('Comparison of Detection Counts by Class')
            
            # x축 레이블 설정 (클래스 이름)
            if len(all_classes) > 10:
                # 클래스가 많으면 일부만 표시
                plt.xticks(x[::2], [class_names[i] for i in range(0, len(class_names), 2)], rotation=45, ha='right')
            else:
                plt.xticks(x, class_names, rotation=45, ha='right')
            
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'class_detection_counts.png', dpi=200)
            
            print(f"클래스별 활성화 패턴 분석 완료. 결과 저장됨: {self.output_dir / 'class_confidence_distribution.png'}")
            
            return class_confs_scratch, class_confs_finetuned
            
        except Exception as e:
            print(f"클래스별 활성화 패턴 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    def _detect_tuned_classes(self, weight_diff=None, top_n=3):
        """
        가중치 변화 분석을 통해 가장 많이 파인튜닝된 클래스를 추정합니다.
        
        Args:
            weight_diff: 미리 계산된 가중치 차이 정보 (없으면 새로 계산)
            top_n: 추정할 상위 클래스 수
            
        Returns:
            list: 파인튜닝된 것으로 추정되는 클래스 인덱스 리스트
        """
        try:
            # 가중치 차이 정보가 없으면 계산
            if weight_diff is None:
                weight_diff = self.compare_weight_distributions()
            if weight_diff is None:
                return []
            
            # 출력 레이어 및 클래스 관련 레이어 찾기
            class_layers = {}
            output_layers = []
            
            # YOLO 모델에서 클래스 관련 레이어는 주로 detection head의 마지막 conv
            # 또는 classifier 레이어에 있음
            for layer_name, diff in weight_diff:
                # 클래스 관련 레이어 찾기 (모델 구조에 따라 다를 수 있음)
                if any(kw in layer_name for kw in ['m.0', 'm.1', 'm.2', 'class', 'cls']):
                    output_layers.append((layer_name, diff))
                    
                    # 클래스 인덱스 추출 시도 (예: 'model.24.m.2' -> 2)
                    parts = layer_name.split('.')
                    for part in parts:
                        if part.isdigit() and int(part) < self.nc:
                            class_idx = int(part)
                            if class_idx not in class_layers or diff > class_layers[class_idx]:
                                class_layers[class_idx] = diff
            
            # 출력 레이어에서 클래스 추출에 실패한 경우, 
            # 출력 레이어의 변화가 큰 레이어를 분석
            if not class_layers and output_layers:
                # 파인튜닝 전 후 모델의 출력 가중치 비교
                scratch_model = self.scratch_model
                finetuned_model = self.finetuned_model
                
                # 가중치 사전 비교
                scratch_weights = scratch_model.state_dict()
                finetuned_weights = finetuned_model.state_dict()
                
                # 모델의 출력 레이어 찾기 (각 모델 구조마다 다름)
                # YOLOv7의 경우 model.24.m.0, model.24.m.1, model.24.m.2에 
                # 각각 box, obj, cls에 대한 가중치가 있음
                output_weight_keys = []
                for key in scratch_weights.keys():
                    if 'weight' in key and any(kw in key for kw in ['m.', 'class', 'cls']):
                        output_weight_keys.append(key)
                
                # 클래스별 가중치 변화 계산
                class_diffs = np.zeros(self.nc)
                for key in output_weight_keys:
                    if key in scratch_weights and key in finetuned_weights:
                        try:
                            # 출력 가중치에서 클래스별 변화 계산
                            s_weight = scratch_weights[key].detach().cpu().numpy()
                            f_weight = finetuned_weights[key].detach().cpu().numpy()
                            
                            if len(s_weight.shape) > 1 and s_weight.shape[0] == self.nc:
                                # 클래스별 가중치 차이 계산
                                diffs = np.mean(np.abs(f_weight - s_weight), axis=tuple(range(1, len(s_weight.shape))))
                                class_diffs += diffs
                        except Exception as e:
                            print(f"클래스 가중치 분석 중 오류: {e}")
                
                # 가중치 변화가 큰 클래스 저장
                for i in range(self.nc):
                    class_layers[i] = class_diffs[i]
            
            # 변화가 큰 순서로 클래스 정렬
            sorted_classes = sorted(class_layers.items(), key=lambda x: x[1], reverse=True)
            
            # 상위 N개 클래스 추출 (또는 변화가 평균보다 큰 클래스)
            tuned_classes = []
            if sorted_classes:
                # 변화량 평균 계산
                avg_diff = np.mean([diff for _, diff in sorted_classes])
                
                # 평균보다 변화가 큰 클래스 또는 상위 N개 클래스 선택
                for cls_idx, diff in sorted_classes:
                    if len(tuned_classes) < top_n or diff > avg_diff:
                        tuned_classes.append(cls_idx)
                    if len(tuned_classes) >= top_n and diff <= avg_diff:
                        break
            
            return sorted(tuned_classes)
        
        except Exception as e:
            print(f"파인튜닝된 클래스 감지 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    def generate_comparison_report(self, test_images=None, report_path=None):
        """두 모델의 종합적인 비교 보고서를 생성합니다."""
        if report_path is None:
            report_path = self.output_dir / 'model_comparison_report.md'
            
        print(f"종합 비교 보고서 생성 중...")
        
        # 데이터로더 생성
        try:
            dataloader = self.create_dataloader()
        except Exception as e:
            print(f"데이터로더 생성 실패: {e}")
            dataloader = None
        
        # 1. 성능 비교
        class_ap_scratch, class_ap_finetuned = self.compare_performance()
        
        # 2. 가중치 분포 비교
        weight_diff = self.compare_weight_distributions()
        
        # 3. 혼동 행렬 분석
        confusion_matrix_scratch, confusion_matrix_finetuned = self.generate_confusion_matrices(dataloader)
        
        # 4. 테스트 이미지가 있는 경우 피처맵 비교
        if test_images and len(test_images) > 0:
            self.compare_feature_maps(test_images[0])
            
            # 5. 클래스별 활성화 패턴 분석
            self.analyze_class_activations(test_images)
        
        # 보고서 생성
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# YOLO 모델 비교 분석 보고서\n\n")
            
            # 모델 정보
            f.write("## 모델 정보\n\n")
            f.write(f"- 스크래치 모델: `{self.scratch_model_path}`\n")
            f.write(f"- 파인튜닝 모델: `{self.finetuned_model_path}`\n")
            f.write(f"- 클래스 수: {self.nc}\n")
            f.write(f"- 클래스 이름: {self.class_names}\n")
            
            tuned_classes = self._detect_tuned_classes(weight_diff)
            if tuned_classes:
                tuned_class_info = ", ".join([f"{cls}번 ({self.class_names[cls] if cls < len(self.class_names) else '알 수 없음'})" for cls in tuned_classes])
                f.write(f"- 추정된 파인튜닝 클래스: {tuned_class_info}\n\n")
            else:
                f.write("- 파인튜닝된 클래스를 추정할 수 없습니다. 가중치 변화가 모든 클래스에 고르게 분포되어 있습니다.\n\n")

            # 테스트 데이터의 클래스 정보 추가
            if class_ap_scratch is not None:
                present_classes = [i for i in range(self.nc) if class_ap_scratch[i] > 0 or class_ap_finetuned[i] > 0]
                if present_classes:
                    present_class_info = ", ".join([f"{cls}번 ({self.class_names[cls] if cls < len(self.class_names) else '알 수 없음'})" for cls in present_classes])
                    f.write(f"- 테스트 데이터에 존재하는 클래스: {present_class_info}\n\n")
                    
                    # 파인튜닝 클래스와 테스트 데이터 클래스의 교집합 확인
                    evaluated_tuned = set(tuned_classes) & set(present_classes)
                    if evaluated_tuned:
                        evaluated_info = ", ".join([f"{cls}번 ({self.class_names[cls] if cls < len(self.class_names) else '알 수 없음'})" for cls in evaluated_tuned])
                        f.write(f"- 평가 가능한 파인튜닝 클래스: {evaluated_info}\n\n")
                    else:
                        f.write("- 주의: 추정된 파인튜닝 클래스 중 테스트 데이터에 존재하는 클래스가 없어 성능 평가가 제한적입니다.\n\n")

            # 성능 비교
            f.write("## 1. 성능 비교\n\n")
            
            if class_ap_scratch is not None and class_ap_finetuned is not None:
                f.write("### 클래스별 AP@0.5 비교\n\n")
                f.write("| 클래스 | 스크래치 모델 | 파인튜닝 모델 | 차이 |\n")
                f.write("|--------|--------------|--------------|------|\n")
                
                for i in range(self.nc):
                    if i < len(class_ap_scratch) and i < len(class_ap_finetuned):
                        scratch_ap = class_ap_scratch[i]
                        finetuned_ap = class_ap_finetuned[i]
                        diff = finetuned_ap - scratch_ap
                        class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                        
                        # 파인튜닝된 클래스 강조
                        if i in [0, 5]:
                            class_name = f"**{class_name} (파인튜닝)**"
                            
                        diff_str = f"{diff:.4f} ({'증가' if diff > 0 else '감소'})"
                        f.write(f"| {class_name} | {scratch_ap:.4f} | {finetuned_ap:.4f} | {diff_str} |\n")
                
                f.write("\n![성능 비교](./performance_comparison.png)\n\n")
            else:
                f.write("성능 비교 데이터를 생성할 수 없습니다.\n\n")
            
            # 가중치 분포 비교
            f.write("## 2. 가중치 변화 분석\n\n")
            
            if weight_diff is not None:
                f.write("### 가중치 차이가 큰 상위 10개 레이어\n\n")
                f.write("| 레이어 | 평균 가중치 차이 |\n")
                f.write("|--------|----------------|\n")
                
                for layer, diff in weight_diff[:10]:
                    # 출력 레이어나 클래스 관련 레이어 강조
                    if any(class_keyword in layer for class_keyword in ['24.m', '23.m', 'cls', 'class']):
                        layer = f"**{layer} (출력/클래스 관련)**"
                    f.write(f"| {layer} | {diff:.6f} |\n")
                
                f.write("\n![가중치 분포 비교](./weight_distribution_comparison.png)\n\n")
            else:
                f.write("가중치 분포 비교 데이터를 생성할 수 없습니다.\n\n")
            
            # 혼동 행렬 분석
            f.write("## 3. 혼동 행렬 분석\n\n")
            
            if confusion_matrix_scratch is not None and confusion_matrix_finetuned is not None:
                f.write("혼동 행렬은 모델이 각 클래스를 얼마나 정확하게 분류하는지 보여줍니다. ")
                f.write("파인튜닝 전후의 혼동 행렬을 비교하면 특정 클래스의 성능 변화를 확인할 수 있습니다.\n\n")
                
                f.write("![혼동 행렬](./confusion_matrices.png)\n\n")
                f.write("![파인튜닝된 클래스 예측 비교](./finetuned_classes_prediction_comparison.png)\n\n")
                
                # 파인튜닝된 클래스에 대한 분석
                f.write("### 파인튜닝된 클래스의 변화\n\n")
                
                tuned_classes = [0, 5]
                for cls_idx in tuned_classes:
                    if cls_idx < self.nc and cls_idx < confusion_matrix_scratch.shape[0]:
                        class_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Class {cls_idx}"
                        
                        # 클래스의 정확한 예측 비율
                        scratch_accuracy = confusion_matrix_scratch[cls_idx, cls_idx] / (confusion_matrix_scratch[cls_idx].sum() + 1e-10)
                        finetuned_accuracy = confusion_matrix_finetuned[cls_idx, cls_idx] / (confusion_matrix_finetuned[cls_idx].sum() + 1e-10)
                        acc_diff = finetuned_accuracy - scratch_accuracy
                        
                        f.write(f"- **{class_name} (인덱스: {cls_idx})**: ")
                        f.write(f"정확도가 {scratch_accuracy:.2%}에서 {finetuned_accuracy:.2%}로 ")
                        
                        if acc_diff > 0:
                            f.write(f"**{acc_diff:.2%} 증가**했습니다.\n")
                        else:
                            f.write(f"**{-acc_diff:.2%} 감소**했습니다.\n")
            else:
                f.write("혼동 행렬 데이터를 생성할 수 없습니다.\n\n")
            
            # 피처맵 및 활성화 패턴 분석
            if test_images and len(test_images) > 0:
                f.write("## 4. 피처맵 비교 분석\n\n")
                f.write("피처맵은 모델의 각 레이어가 이미지의 어떤 특징에 반응하는지 보여줍니다. ")
                f.write("파인튜닝 전후의 피처맵을 비교하면 모델이 특정 클래스의 특징을 어떻게 다르게 인식하는지 확인할 수 있습니다.\n\n")
                
                f.write("![피처맵 비교](./feature_map_comparison.png)\n\n")
                
                f.write("## 5. 클래스별 활성화 패턴 분석\n\n")
                f.write("활성화 패턴 분석은 모델이 각 클래스에 대해 얼마나 확신을 가지고 예측하는지 보여줍니다. ")
                f.write("신뢰도 분포 비교를 통해 파인튜닝이 모델의 신뢰도에 미치는 영향을 파악할 수 있습니다.\n\n")
                
                f.write("![클래스별 신뢰도 분포](./class_confidence_distribution.png)\n\n")
                f.write("![클래스별 검출 횟수](./class_detection_counts.png)\n\n")
                
                # 파인튜닝된 클래스에 대한 분석
                f.write("### 파인튜닝된 클래스의 활성화 패턴 변화\n\n")
                f.write("파인튜닝을 통해 특정 클래스에 대한 모델의 활성화 패턴이 어떻게 변했는지 분석합니다.\n\n")
                f.write("- 파인튜닝된 클래스는 일반적으로 더 높은 신뢰도 값을 가지는 경향이 있습니다.\n")
                f.write("- 높은 신뢰도는 모델이 해당 클래스를 더 확실하게 인식한다는 것을 의미합니다.\n")
            
            # 결론
            f.write("## 결론\n\n")
            if tuned_classes:
                tuned_class_info = ", ".join([f"{cls}번" for cls in tuned_classes])
                f.write(f"이 분석을 통해 특정 클래스({tuned_class_info})에 대한 파인튜닝이 모델 성능에 미치는 영향을 확인했습니다.\n\n")
            else:
                f.write("이 분석을 통해 모델 전체에 대한 파인튜닝이 성능에 미치는 영향을 확인했습니다.\n\n")
            
            # 주요 발견 사항 (데이터 기반으로 동적 생성)
            f.write("### 주요 발견 사항\n\n")
            
            # 성능 향상 여부 (AP 기준)
            if class_ap_scratch is not None and class_ap_finetuned is not None:
            # 하드코딩된 tuned_classes = [0, 5] 삭제
                # 감지된 클래스 사용
                tuned_classes_in_test = [cls for cls in tuned_classes if cls < len(class_ap_scratch)]
                
                if tuned_classes_in_test:
                    tuned_improved = sum(1 for idx in tuned_classes_in_test if class_ap_finetuned[idx] > class_ap_scratch[idx])
                    
                    if tuned_improved == len(tuned_classes_in_test):
                        f.write("- **모든 파인튜닝된 클래스의 성능이 향상**되었습니다.\n")
                    elif tuned_improved > 0:
                        f.write(f"- **일부 파인튜닝된 클래스({tuned_improved}/{len(tuned_classes_in_test)})의 성능이 향상**되었습니다.\n")
                    else:
                        f.write("- 파인튜닝된 클래스의 성능 향상이 관찰되지 않았습니다.\n")
                
                # 다른 클래스에 미치는 영향
                other_classes = [i for i in range(self.nc) if i not in tuned_classes_in_test 
                                and i < len(class_ap_scratch) 
                                and (class_ap_scratch[i] > 0 or class_ap_finetuned[i] > 0)]
                
                if other_classes:
                    other_improved = sum(1 for idx in other_classes if class_ap_finetuned[idx] > class_ap_scratch[idx])
                    other_degraded = sum(1 for idx in other_classes if class_ap_finetuned[idx] < class_ap_scratch[idx])
                    
                    if other_degraded > len(other_classes) / 2:
                        f.write(f"- **대부분의 다른 클래스({other_degraded}/{len(other_classes)})에서 성능 저하**가 발생했습니다. (Catastrophic forgetting)\n")
                    elif other_improved > len(other_classes) / 2:
                        f.write(f"- **대부분의 다른 클래스({other_improved}/{len(other_classes)})에서도 성능 향상**이 관찰되었습니다. (Positive transfer)\n")
                    else:
                        f.write(f"- 다른 클래스에 미치는 영향은 혼합되어 있습니다. ({other_improved} 향상, {other_degraded} 저하)\n")
            
            # 가중치 분포 관련 발견
            if weight_diff is not None:
                output_layers_changed = sum(1 for layer, _ in weight_diff[:20] 
                                          if any(kw in layer for kw in ['23.m', '24.m', 'cls', 'class']))
                
                if output_layers_changed > 5:
                    f.write("- **출력 레이어(클래스 감지 관련)의 가중치가 크게 변화**했습니다.\n")
                else:
                    f.write("- 주로 중간 레이어의 가중치가 변화했습니다.\n")
            
            # 추가 권장 사항
            f.write("\n### 권장 사항\n\n")
            f.write("1. **파인튜닝 전략 조정**: ")
            
            # 성능 기반 권장사항
            if class_ap_scratch is not None and class_ap_finetuned is not None:
                if tuned_improved < len(tuned_classes):
                    f.write("일부 파인튜닝된 클래스의 성능이 향상되지 않았으므로, 더 많은 데이터나 다른 하이퍼파라미터 설정을 시도해보세요.\n")
                elif other_degraded > len(other_classes) / 2:
                    f.write("다른 클래스의 성능 저하를 방지하기 위해 전체 데이터셋에서 샘플링한 이미지를 함께 사용하는 방식으로 파인튜닝하세요.\n")
                else:
                    f.write("현재 파인튜닝 전략이 효과적이므로 계속 사용하세요.\n")
            else:
                f.write("더 정확한 성능 평가를 위해 더 큰 테스트 데이터셋으로 검증해보세요.\n")
            
            f.write("2. **특정 클래스 임계값 조정**: ")
            f.write("파인튜닝된 클래스의 신뢰도 임계값을 조정하여 정밀도와 재현율의 균형을 최적화하세요.\n")
            
            f.write("3. **추가 파인튜닝 방향**: ")
            f.write("이 분석 결과를 토대로 추가 훈련이 필요한 클래스를 식별하고 집중적으로 데이터를 수집하세요.\n")
            
        print(f"종합 비교 보고서가 생성되었습니다: {report_path}")
        
        return report_path
    
    def run_all_analyses(self, test_images=None):
        """모든 분석을 순차적으로 실행합니다."""
        # 1. 성능 비교
        print("\n=== 1. 성능 비교 분석 ===")
        self.compare_performance()
        
        # 2. 가중치 분포 비교
        print("\n=== 2. 가중치 분포 비교 분석 ===")
        self.compare_weight_distributions()
        
        # 3. 혼동 행렬 분석
        print("\n=== 3. 혼동 행렬 분석 ===")
        self.generate_confusion_matrices()
        
        # 4. 테스트 이미지가 있는 경우 추가 분석
        if test_images and len(test_images) > 0:
            # 4.1 피처맵 비교
            print("\n=== 4. 피처맵 비교 분석 ===")
            self.compare_feature_maps(test_images[0])
            
            # 4.2 클래스별 활성화 패턴 분석
            print("\n=== 5. 클래스별 활성화 패턴 분석 ===")
            self.analyze_class_activations(test_images)
        
        # 5. 종합 보고서 생성
        print("\n=== 6. 종합 보고서 생성 ===")
        report_path = self.generate_comparison_report(test_images)
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='YOLO 모델 비교 분석 도구')
    parser.add_argument('--scratch', type=str,default='d:\\GIT_DOWN\\yolov7_1\\guyoung\\safe-v4.pt', help='스크래치 모델 경로')
    parser.add_argument('--finetuned', type=str, default='d:\\GIT_DOWN\\yolov7_1\\guyoung\\guyoung.pt', help='파인튜닝 모델 경로')
    parser.add_argument('--data', type=str, default='z:\\101.etc\\guyoung\\data\\list\\2\\data_test.yaml',help='데이터 설정 YAML 경로')
    parser.add_argument('--img-size', type=int, default=640, help='이미지 크기')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU 임계값')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--device', type=str, default='0', help='장치 선택 (예: 0, 0,1,2,3, cpu)')
    parser.add_argument('--test-images', type=str, nargs='+', default=[], help='테스트 이미지 경로 (여러 개 가능)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--hyp', type=str, default='Z:\\101.etc\\guyoung\\file\\data\\hyp.scratch.custom.yaml', help='hyperparameters path')

    
    args = parser.parse_args()

    # 모델 비교 분석 실행
    model_comparer = ModelComparer(
        scratch_model_path=args.scratch,
        finetuned_model_path=args.finetuned,
        data_yaml=args.data,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        batch_size=args.batch_size,
        device=args.device,
        opt=args
    )
    
    # 모든 분석 실행
    report_path = model_comparer.run_all_analyses(args.test_images)
    print(f"\n분석이 완료되었습니다. 결과를 확인하세요: {report_path}")

if __name__ == '__main__':
    main()
    
