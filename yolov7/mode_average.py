import torch
import argparse
import numpy as np

def average_weights(model_paths, output_path, weights=None):
    """
    여러 모델의 가중치를 평균하여 새로운 모델을 생성합니다.
    
    Args:
        model_paths (list): 평균화할 모델 가중치 파일 경로 목록 (.pt)
        output_path (str): 평균화된 모델 가중치를 저장할 파일 경로 (.pt)
        weights (list, optional): 각 모델의 가중치 비율. None이면 동일한 비율로 평균화합니다.
    """
    num_models = len(model_paths)
    
    if weights is None:
        # 모든 모델에 동일한 가중치 적용
        weights = [1.0 / num_models] * num_models
    else:
        # 가중치 정규화하여 합이 1이 되도록 함
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    print(f"총 {num_models}개 모델의 가중치를 평균화합니다.")
    
    # 첫 번째 모델을 템플릿으로 사용
    print(f"템플릿 모델 로드 중: {model_paths[0]}")
    template_ckpt = torch.load(model_paths[0], map_location='cpu')
    
    # 모델 구조 확인 및 state_dict 추출
    if 'model' in template_ckpt:
        template_model = template_ckpt['model'].float()
        averaged_state_dict = template_model.state_dict()
        model_key = 'model'
    elif 'state_dict' in template_ckpt:
        averaged_state_dict = template_ckpt['state_dict']
        model_key = 'state_dict'
    else:
        averaged_state_dict = template_ckpt
        model_key = None
    
    # 첫 번째 모델의 가중치에 첫 번째 가중치 계수 적용
    for key in averaged_state_dict:
        averaged_state_dict[key] = averaged_state_dict[key] * weights[0]
    
    # 나머지 모델들의 가중치를 누적하여 더함
    for i in range(1, num_models):
        print(f"모델 로드 중 ({i+1}/{num_models}): {model_paths[i]}")
        ckpt = torch.load(model_paths[i], map_location='cpu')
        
        if model_key == 'model' and 'model' in ckpt:
            state_dict = ckpt['model'].float().state_dict()
        elif model_key == 'state_dict' and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif model_key is None:
            state_dict = ckpt
        else:
            print(f"경고: 모델 {i+1}의 구조가 템플릿과 다릅니다.")
            continue
        
        # 각 레이어의 가중치를 가중 평균
        for key in averaged_state_dict:
            if key in state_dict:
                averaged_state_dict[key] += state_dict[key] * weights[i]
            else:
                print(f"경고: 키 '{key}'가 모델 {i+1}에 없습니다.")
    
    # 평균화된 가중치로 새 모델 생성
    output_ckpt = template_ckpt.copy()
    if model_key == 'model':
        template_model.load_state_dict(averaged_state_dict)
        output_ckpt['model'] = template_model
    elif model_key == 'state_dict':
        output_ckpt['state_dict'] = averaged_state_dict
    else:
        output_ckpt = averaged_state_dict
    
    # 메타데이터 추가
    if isinstance(output_ckpt, dict):
        output_ckpt['average_weights_info'] = {
            'source_models': model_paths,
            'weights': weights
        }
    
    # 평균화된 모델 저장
    print(f"평균화된 모델 저장 중: {output_path}")
    torch.save(output_ckpt, output_path)
    print(f"모델이 성공적으로 평균화되어 {output_path}에 저장되었습니다.")
    print(f"적용된 가중치 비율: {weights}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 가중치 평균화 도구")
    parser.add_argument('--models', nargs='+', type=str, required=True, 
                        help='평균화할 모델 가중치 파일 경로들 (공백으로 구분)')
    parser.add_argument('--output', type=str, required=True, 
                        help='평균화된 모델을 저장할 경로')
    parser.add_argument('--weights', nargs='+', type=float, 
                        help='각 모델의 가중치 비율 (공백으로 구분, 설정하지 않으면 균등 비율 적용)')
    
    args = parser.parse_args()
    
    if args.weights and len(args.weights) != len(args.models):
        print(f"오류: 모델 수({len(args.models)})와 가중치 수({len(args.weights)})가 일치하지 않습니다.")
        exit(1)
    
    average_weights(
        args.models, 
        args.output, 
        args.weights
    )