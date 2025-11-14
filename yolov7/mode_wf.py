import torch
import argparse
import numpy as np

def merge_weights(scratch_model_path, finetuned_model_path, output_path, alpha=0.5):
    """
    WiSE-FT 방식으로 YOLOv7 모델의 가중치를 보간하여 새로운 모델을 생성합니다.
    
    Args:
        scratch_model_path (str): 제로샷(scratch) 모델 가중치 파일 경로 (.pt)
        finetuned_model_path (str): 파인튜닝된 모델 가중치 파일 경로 (.pt)
        output_path (str): 보간된 모델 가중치를 저장할 파일 경로 (.pt)
        alpha (float): 가중치 보간 계수 (0~1 사이, 기본값: 0.5)
    """
    print(f"제로샷 모델 로드 중: {scratch_model_path}")
    scratch_ckpt = torch.load(scratch_model_path, map_location='cpu')
    scratch_state_dict = scratch_ckpt['model'].float().state_dict() if 'model' in scratch_ckpt else scratch_ckpt['state_dict'] if 'state_dict' in scratch_ckpt else scratch_ckpt
    
    print(f"파인튜닝 모델 로드 중: {finetuned_model_path}")
    finetuned_ckpt = torch.load(finetuned_model_path, map_location='cpu')
    finetuned_state_dict = finetuned_ckpt['model'].float().state_dict() if 'model' in finetuned_ckpt else finetuned_ckpt['state_dict'] if 'state_dict' in finetuned_ckpt else finetuned_ckpt
    
    # WiSE-FT 방식의 가중치 보간을 위한 새 state_dict 생성
    wise_ft_state_dict = {}
    
    # 두 모델의 구조가 동일한지 확인
    if scratch_state_dict.keys() != finetuned_state_dict.keys():
        print("경고: 두 모델의 구조가 다릅니다. 교차하는 레이어만 보간합니다.")
    
    # 가중치 선형 보간 (WiSE-FT 방식)
    for key in scratch_state_dict.keys():
        if key in finetuned_state_dict:
            # WiSE-FT 공식: θα = (1 - α) · θzero-shot + α · θfine-tuned
            wise_ft_state_dict[key] = (1 - alpha) * scratch_state_dict[key] + alpha * finetuned_state_dict[key]
        else:
            print(f"경고: 키 '{key}'가 파인튜닝 모델에 없습니다. 제로샷 모델의 가중치만 사용합니다.")
            wise_ft_state_dict[key] = scratch_state_dict[key]
    
    # 보간된 가중치로 새 모델 생성
    if 'model' in scratch_ckpt:
        scratch_ckpt['model'].float().load_state_dict(wise_ft_state_dict)
        output_ckpt = scratch_ckpt
    else:
        output_ckpt = {'state_dict': wise_ft_state_dict}
        # 추가 메타데이터 복사
        for key in scratch_ckpt:
            if key != 'state_dict' and key != 'model':
                output_ckpt[key] = scratch_ckpt[key]
    
    # WiSE-FT 메타데이터 추가
    output_ckpt['wise_ft_info'] = {
        'zero_shot_model': scratch_model_path,
        'fine_tuned_model': finetuned_model_path,
        'alpha': alpha,
        'interpolation_description': 'Weight-space Ensemble Fine-Tuning (WiSE-FT)'
    }
    
    # 보간된 모델 저장
    print(f"WiSE-FT 보간 모델 저장 중: {output_path}")
    torch.save(output_ckpt, output_path)
    print(f"모델이 WiSE-FT 방식으로 성공적으로 보간되어 {output_path}에 저장되었습니다.")
    print(f"보간 계수 α: {alpha}")

def calculate_weight_difference(scratch_model_path, finetuned_model_path):
    """두 모델 간 가중치 차이를 분석합니다."""
    scratch_ckpt = torch.load(scratch_model_path, map_location='cpu')
    finetuned_ckpt = torch.load(finetuned_model_path, map_location='cpu')
    
    scratch_state_dict = scratch_ckpt['model'].float().state_dict() if 'model' in scratch_ckpt else scratch_ckpt['state_dict'] if 'state_dict' in scratch_ckpt else scratch_ckpt
    finetuned_state_dict = finetuned_ckpt['model'].float().state_dict() if 'model' in finetuned_ckpt else finetuned_ckpt['state_dict'] if 'state_dict' in finetuned_ckpt else finetuned_ckpt
    
    weight_differences = {}
    total_diff = 0
    
    for key in scratch_state_dict.keys():
        if key in finetuned_state_dict:
            # 절대 가중치 차이 계산
            diff = torch.abs(scratch_state_dict[key] - finetuned_state_dict[key])
            weight_differences[key] = diff.mean().item()
            total_diff += diff.mean().item()
    
    # 상위 10개 변화가 큰 레이어 출력
    sorted_differences = sorted(weight_differences.items(), key=lambda x: x[1], reverse=True)
    print("\n가장 많이 변경된 상위 10개 레이어:")
    for layer, diff in sorted_differences[:10]:
        print(f"{layer}: {diff}")
    
    print(f"\n전체 평균 가중치 변화: {total_diff / len(weight_differences)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiSE-FT 모델 가중치 보간 도구")
    parser.add_argument('--scratch', type=str, required=True, help='제로샷 모델 가중치 파일 경로')
    parser.add_argument('--finetuned', type=str, required=True, help='파인튜닝 모델 가중치 파일 경로')
    parser.add_argument('--output', type=str, required=True, help='보간된 모델을 저장할 경로')
    parser.add_argument('--alpha', type=float, default=0.5, help='가중치 보간 계수 (기본값: 0.5)')
    parser.add_argument('--analyze', action='store_true', help='모델 가중치 차이 분석')
    
    args = parser.parse_args()
    
    if args.analyze:
        # 모델 가중치 차이 분석
        calculate_weight_difference(args.scratch, args.finetuned)
    else:
        # 모델 가중치 보간
        merge_weights(
            args.scratch, 
            args.finetuned, 
            args.output, 
            args.alpha
        )