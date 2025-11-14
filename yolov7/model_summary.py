import torch
from models.experimental import attempt_load
import matplotlib.pyplot as plt
import numpy as np

# YOLOv7 모델 로드
model_path = 'best.pt'  # 모델 파일 경로
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(model_path, map_location=device)
model = attempt_load(model_path, map_location=device)

print("===== 모델 기본 정보 =====")
print(f"모델 이름: {type(model).__name__}")
print(f"모델 클래스: {type(model)}")

# 체크포인트에서 학습 정보 추출
print("\n===== 학습 정보 =====")
if 'epoch' in ckpt:
    print(f"학습된 에폭: {ckpt['epoch']}")
if 'best_fitness' in ckpt:
    print(f"최고 적합도: {ckpt['best_fitness']}")
if 'training_results' in ckpt:
    print(f"학습 결과 요약:\n{ckpt['training_results']}")
if 'optimizer' in ckpt:
    print("\n최적화기 정보:")
    optimizer = ckpt['optimizer']
    print(f"최적화기 유형: {type(optimizer).__name__}")
    if hasattr(optimizer, 'param_groups'):
        for i, group in enumerate(optimizer.param_groups):
            print(f"\n파라미터 그룹 {i}:")
            for key in group:
                if key != 'params':
                    print(f"  {key}: {group[key]}")

# 학습 이력 정보 (있는 경우)
if 'scheduler' in ckpt:
    print("\n스케줄러 정보:")
    print(f"스케줄러 유형: {type(ckpt['scheduler']).__name__}")
    
# 손실 함수 이력 시각화 (있는 경우)
if 'results_dict' in ckpt:
    print("\n손실 이력:")
    results = ckpt['results_dict']
    for key, value in results.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"{key}: {value}")
            plt.figure(figsize=(10, 5))
            plt.plot(value)
            plt.title(f'{key} 변화')
            plt.xlabel('에폭')
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig(f'{key}_history.png')
            print(f"{key} 그래프가 {key}_history.png 파일로 저장되었습니다.")

# 하이퍼파라미터 정보
if 'hyp' in ckpt:
    print("\n하이퍼파라미터:")
    for key, value in ckpt['hyp'].items():
        print(f"  {key}: {value}")

# 데이터셋 정보
if 'dataset' in ckpt:
    print("\n데이터셋 정보:")
    print(ckpt['dataset'])

# 클래스 정보
if hasattr(model, 'names'):
    print("\n클래스 이름:")
    for i, name in enumerate(model.names):
        print(f"  클래스 {i}: {name}")

# 모델 구성 정보 (YOLOv7은 yaml 구성을 사용함)
if 'model_cfg' in ckpt:
    print("\n모델 구성 파일:")
    print(ckpt['model_cfg'])

# 학습 설정
if 'opt' in ckpt:
    print("\n학습 옵션:")
    for key, value in ckpt['opt'].items():
        print(f"  {key}: {value}")

# 학습 날짜 및 시간
if 'date' in ckpt:
    print(f"\n학습 날짜: {ckpt['date']}")

# 추가 파라미터 정보
print("\n===== 모델 파라미터 통계 =====")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"총 파라미터 개수: {total_params:,}")
print(f"학습 가능한 파라미터 개수: {trainable_params:,}")
print(f"학습 불가능한 파라미터 개수: {total_params - trainable_params:,}")