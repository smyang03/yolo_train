import torch
import argparse

def merge_weights(scratch_model_path, finetuned_model_path, output_path, scratch_weight=0.9, finetuned_weight=0.1):
    """
    YOLOv7 모델의 가중치를 혼합하여 새로운 모델을 생성합니다.
    
    Args:
        scratch_model_path (str): scratch로 학습된 모델 가중치 파일 경로 (.pt)
        finetuned_model_path (str): 파인튜닝된 모델 가중치 파일 경로 (.pt)
        output_path (str): 혼합된 모델 가중치를 저장할 파일 경로 (.pt)
        scratch_weight (float): scratch 모델 가중치 비율 (기본값: 0.9)
        finetuned_weight (float): 파인튜닝 모델 가중치 비율 (기본값: 0.1)
    """
    print(f"스크래치 모델 로드 중: {scratch_model_path}")
    scratch_ckpt = torch.load(scratch_model_path, map_location='cpu')
    scratch_state_dict = scratch_ckpt['model'].float().state_dict() if 'model' in scratch_ckpt else scratch_ckpt['state_dict'] if 'state_dict' in scratch_ckpt else scratch_ckpt
    
    print(f"파인튜닝 모델 로드 중: {finetuned_model_path}")
    finetuned_ckpt = torch.load(finetuned_model_path, map_location='cpu')
    finetuned_state_dict = finetuned_ckpt['model'].float().state_dict() if 'model' in finetuned_ckpt else finetuned_ckpt['state_dict'] if 'state_dict' in finetuned_ckpt else finetuned_ckpt
    
    # 혼합된 가중치를 저장할 새 state_dict 생성
    merged_state_dict = {}
    
    # 두 모델의 구조가 동일한지 확인
    if scratch_state_dict.keys() != finetuned_state_dict.keys():
        print("경고: 두 모델의 구조가 다릅니다. 교차하는 레이어만 병합합니다.")
    
    # 가중치 혼합
    for key in scratch_state_dict.keys():
        if key in finetuned_state_dict:
            # 두 모델의 가중치에 각각 지정된 비율을 적용하여 혼합
            merged_state_dict[key] = scratch_state_dict[key] * scratch_weight + finetuned_state_dict[key] * finetuned_weight
        else:
            print(f"경고: 키 '{key}'가 파인튜닝 모델에 없습니다. scratch 모델의 가중치만 사용합니다.")
            merged_state_dict[key] = scratch_state_dict[key]
    
    # 혼합된 가중치로 새 모델 생성
    if 'model' in scratch_ckpt:
        scratch_ckpt['model'].float().load_state_dict(merged_state_dict)
        output_ckpt = scratch_ckpt
    else:
        output_ckpt = {'state_dict': merged_state_dict}
        # 추가 메타데이터 복사
        for key in scratch_ckpt:
            if key != 'state_dict' and key != 'model':
                output_ckpt[key] = scratch_ckpt[key]
    
    # 혼합 메타데이터 추가
    output_ckpt['merged_info'] = {
        'scratch_model': scratch_model_path,
        'finetuned_model': finetuned_model_path,
        'scratch_weight': scratch_weight,
        'finetuned_weight': finetuned_weight
    }
    
    # 혼합된 모델 저장
    print(f"혼합된 모델 저장 중: {output_path}")
    torch.save(output_ckpt, output_path)
    print(f"모델이 성공적으로 혼합되어 {output_path}에 저장되었습니다.")
    print(f"가중치 비율 - 스크래치: {scratch_weight}, 파인튜닝: {finetuned_weight}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv7 모델 가중치 혼합 도구")
    parser.add_argument('--scratch', type=str, required=True, help='스크래치 모델 가중치 파일 경로')
    parser.add_argument('--finetuned', type=str, required=True, help='파인튜닝 모델 가중치 파일 경로')
    parser.add_argument('--output', type=str, required=True, help='혼합된 모델을 저장할 경로')
    parser.add_argument('--scratch-weight', type=float, default=0.9, help='스크래치 모델 가중치 비율 (기본값: 0.9)')
    parser.add_argument('--finetuned-weight', type=float, default=0.1, help='파인튜닝 모델 가중치 비율 (기본값: 0.1)')
    
    args = parser.parse_args()
    
    merge_weights(
        args.scratch, 
        args.finetuned, 
        args.output, 
        args.scratch_weight, 
        args.finetuned_weight
    )