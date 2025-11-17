"""
src/utils/validation.py
입력 검증 유틸리티 모듈
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Tuple, List, Any


class ConfigValidator:
    """설정 검증 클래스"""

    @staticmethod
    def validate_dataset_path(path: str) -> Tuple[bool, str]:
        """데이터셋 경로 검증

        Args:
            path: 데이터셋 YAML 파일 경로

        Returns:
            (성공 여부, 메시지)
        """
        if not path:
            return False, "데이터셋 경로가 비어있습니다."

        path_obj = Path(path)

        if not path_obj.exists():
            return False, f"데이터셋 파일을 찾을 수 없습니다: {path}"

        if not path_obj.suffix in ['.yaml', '.yml']:
            return False, "데이터셋 파일은 YAML 형식이어야 합니다."

        # YAML 파일 파싱 시도
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 필수 키 확인
            required_keys = ['train', 'val', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in data]

            if missing_keys:
                return False, f"데이터셋 YAML에 필수 키가 누락되었습니다: {', '.join(missing_keys)}"

            # 클래스 수 검증
            nc = data['nc']
            names = data['names']

            if not isinstance(nc, int) or nc <= 0:
                return False, f"잘못된 클래스 수: {nc}"

            if len(names) != nc:
                return False, f"클래스 이름 개수({len(names)})와 클래스 수({nc})가 일치하지 않습니다."

            return True, "데이터셋 검증 성공"

        except yaml.YAMLError as e:
            return False, f"YAML 파일 파싱 오류: {e}"
        except Exception as e:
            return False, f"데이터셋 검증 오류: {e}"

    @staticmethod
    def validate_model_config(path: str) -> Tuple[bool, str]:
        """모델 설정 파일 검증

        Args:
            path: 모델 config YAML 파일 경로

        Returns:
            (성공 여부, 메시지)
        """
        if not path:
            return False, "모델 설정 경로가 비어있습니다."

        path_obj = Path(path)

        if not path_obj.exists():
            return False, f"모델 설정 파일을 찾을 수 없습니다: {path}"

        if not path_obj.suffix in ['.yaml', '.yml']:
            return False, "모델 설정 파일은 YAML 형식이어야 합니다."

        # YAML 파일 파싱 시도
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 기본 구조 확인
            if not isinstance(data, dict):
                return False, "잘못된 모델 설정 형식입니다."

            # YOLOv7 모델 설정 필수 키 확인 (유연하게)
            if 'nc' not in data and 'backbone' not in data and 'head' not in data:
                return False, "올바른 YOLOv7 모델 설정 파일이 아닙니다."

            return True, "모델 설정 검증 성공"

        except yaml.YAMLError as e:
            return False, f"YAML 파일 파싱 오류: {e}"
        except Exception as e:
            return False, f"모델 설정 검증 오류: {e}"

    @staticmethod
    def validate_weights_path(path: str) -> Tuple[bool, str]:
        """가중치 파일 검증

        Args:
            path: 가중치 파일 경로

        Returns:
            (성공 여부, 메시지)
        """
        if not path:
            return True, "가중치 파일 없음 (처음부터 학습)"

        path_obj = Path(path)

        if not path_obj.exists():
            return False, f"가중치 파일을 찾을 수 없습니다: {path}"

        if not path_obj.suffix == '.pt':
            return False, "가중치 파일은 .pt 형식이어야 합니다."

        # 파일 크기 확인 (최소 1MB)
        file_size = path_obj.stat().st_size
        if file_size < 1024 * 1024:  # 1MB
            return False, f"가중치 파일이 너무 작습니다: {file_size / 1024:.1f} KB"

        return True, f"가중치 파일 검증 성공 ({file_size / (1024*1024):.1f} MB)"

    @staticmethod
    def validate_training_params(config: Dict[str, Any]) -> Tuple[bool, str]:
        """훈련 파라미터 검증

        Args:
            config: 훈련 설정 딕셔너리

        Returns:
            (성공 여부, 메시지)
        """
        errors = []

        # Epochs 검증
        epochs = config.get('epochs', 0)
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append(f"잘못된 Epochs 값: {epochs} (양의 정수여야 함)")
        elif epochs > 10000:
            errors.append(f"Epochs 값이 너무 큽니다: {epochs} (권장: 1-1000)")

        # Batch size 검증
        batch_size = config.get('batch_size', 0)
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append(f"잘못된 Batch Size 값: {batch_size} (양의 정수여야 함)")
        elif batch_size > 128:
            errors.append(f"Batch Size가 너무 큽니다: {batch_size} (메모리 부족 가능성)")

        # Image size 검증
        image_size = config.get('image_size', 0)
        valid_sizes = [320, 416, 512, 640, 768, 896, 1024, 1280]
        if image_size not in valid_sizes:
            errors.append(f"지원되지 않는 Image Size: {image_size} (권장: {valid_sizes})")

        # Learning rate 검증
        lr = config.get('learning_rate', 0.0)
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append(f"잘못된 Learning Rate 값: {lr} (양수여야 함)")
        elif lr > 1.0:
            errors.append(f"Learning Rate가 너무 큽니다: {lr} (일반적으로 0.0001-0.1)")

        # Workers 검증
        workers = config.get('workers', 0)
        if not isinstance(workers, int) or workers < 0:
            errors.append(f"잘못된 Workers 값: {workers} (0 이상의 정수여야 함)")
        elif workers > 32:
            errors.append(f"Workers 수가 너무 많습니다: {workers} (권장: 0-16)")

        # Device 검증
        device = config.get('device', '')
        if not device:
            errors.append("Device 값이 비어있습니다.")
        elif not (device == 'cpu' or device.isdigit() or ',' in device):
            errors.append(f"잘못된 Device 값: {device} (예: '0', 'cpu', '0,1')")

        if errors:
            return False, "\n".join(errors)

        return True, "훈련 파라미터 검증 성공"

    @staticmethod
    def validate_hyperparams_file(path: str) -> Tuple[bool, str]:
        """하이퍼파라미터 파일 검증

        Args:
            path: 하이퍼파라미터 YAML 파일 경로

        Returns:
            (성공 여부, 메시지)
        """
        if not path:
            return True, "하이퍼파라미터 파일 없음 (기본값 사용)"

        path_obj = Path(path)

        if not path_obj.exists():
            return False, f"하이퍼파라미터 파일을 찾을 수 없습니다: {path}"

        if not path_obj.suffix in ['.yaml', '.yml']:
            return False, "하이퍼파라미터 파일은 YAML 형식이어야 합니다."

        # YAML 파일 파싱 시도
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                return False, "잘못된 하이퍼파라미터 형식입니다."

            # 주요 하이퍼파라미터 키 확인 (선택적)
            recommended_keys = ['lr0', 'momentum', 'weight_decay', 'warmup_epochs']
            missing_keys = [key for key in recommended_keys if key not in data]

            if missing_keys:
                return True, f"일부 권장 파라미터가 누락되었습니다: {', '.join(missing_keys)} (계속 진행 가능)"

            return True, "하이퍼파라미터 파일 검증 성공"

        except yaml.YAMLError as e:
            return False, f"YAML 파일 파싱 오류: {e}"
        except Exception as e:
            return False, f"하이퍼파라미터 검증 오류: {e}"

    @staticmethod
    def validate_all_settings(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """전체 설정 검증

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            (성공 여부, 오류/경고 메시지 리스트)
        """
        messages = []
        all_valid = True

        # 데이터셋 경로 검증
        if 'dataset_path' in config:
            valid, msg = ConfigValidator.validate_dataset_path(config['dataset_path'])
            if not valid:
                all_valid = False
                messages.append(f"❌ 데이터셋: {msg}")
            else:
                messages.append(f"✅ 데이터셋: {msg}")

        # 모델 설정 검증
        if 'model_config' in config:
            valid, msg = ConfigValidator.validate_model_config(str(config['model_config']))
            if not valid:
                all_valid = False
                messages.append(f"❌ 모델 설정: {msg}")
            else:
                messages.append(f"✅ 모델 설정: {msg}")

        # 가중치 파일 검증
        if 'weights_path' in config and config['weights_path']:
            valid, msg = ConfigValidator.validate_weights_path(config['weights_path'])
            if not valid:
                all_valid = False
                messages.append(f"❌ 가중치: {msg}")
            else:
                messages.append(f"✅ 가중치: {msg}")

        # 훈련 파라미터 검증
        valid, msg = ConfigValidator.validate_training_params(config)
        if not valid:
            all_valid = False
            messages.append(f"❌ 훈련 파라미터:\n{msg}")
        else:
            messages.append(f"✅ 훈련 파라미터: {msg}")

        # 하이퍼파라미터 파일 검증
        if 'hyperparams_file' in config and config.get('hyperparams_file'):
            valid, msg = ConfigValidator.validate_hyperparams_file(config['hyperparams_file'])
            if not valid:
                all_valid = False
                messages.append(f"❌ 하이퍼파라미터: {msg}")
            else:
                messages.append(f"✅ 하이퍼파라미터: {msg}")

        return all_valid, messages


def validate_path_exists(path: str, description: str = "Path") -> Tuple[bool, str]:
    """경로 존재 여부 검증"""
    if not path:
        return False, f"{description}가 비어있습니다."

    if not os.path.exists(path):
        return False, f"{description}를 찾을 수 없습니다: {path}"

    return True, f"{description} 확인됨"


def validate_file_extension(path: str, allowed_extensions: List[str]) -> Tuple[bool, str]:
    """파일 확장자 검증"""
    if not path:
        return False, "파일 경로가 비어있습니다."

    ext = Path(path).suffix.lower()
    if ext not in allowed_extensions:
        return False, f"허용되지 않는 파일 형식: {ext} (허용: {', '.join(allowed_extensions)})"

    return True, "파일 형식 확인됨"


# 사용 예시
if __name__ == "__main__":
    # 테스트
    validator = ConfigValidator()

    # 샘플 설정
    test_config = {
        'dataset_path': 'data/coco.yaml',
        'model_config': 'cfg/training/yolov7.yaml',
        'epochs': 300,
        'batch_size': 16,
        'image_size': 640,
        'learning_rate': 0.01,
        'workers': 8,
        'device': '0'
    }

    valid, messages = validator.validate_all_settings(test_config)

    safe_print("=" * 50)
    safe_print("설정 검증 결과:")
    safe_print("=" * 50)
    for msg in messages:
        safe_print(msg)
    safe_print("=" * 50)
    safe_print(f"전체 검증 결과: {'성공' if valid else '실패'}")
