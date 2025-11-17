"""
from utils import safe_print
설정 관리 모듈 (Python 3.8+ 호환) - 경로 문제 해결
"""

import sys
import io
import yaml
from pathlib import Path
from typing import Dict, Any
import os

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == 'win32':
    try:
        if sys.version_info >= (3, 7):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

class ConfigManager:
    """설정 관리 클래스"""

    def __init__(self):
        # PyInstaller 환경 감지
        if getattr(sys, 'frozen', False):
            self.app_dir = Path(sys.executable).parent
        else:
            self.app_dir = Path(__file__).parent.parent.parent

        self.config_dir = self.app_dir / "resources" / "configs"
        self.default_config_path = self.config_dir / "default.yaml"

        # YOLOv7 경로는 환경 변수 또는 상대 경로로 설정
        if os.environ.get('YOLOV7_PATH'):
            self.yolo_dir = Path(os.environ['YOLOV7_PATH'])
        else:
            self.yolo_dir = self.app_dir.parent / "yolov7"

        self.config = self.load_default_config()
    
    def load_default_config(self) -> Dict[str, Any]:
        """기본 설정 로드"""
        default_config = {
            'dataset': {
                'mode': 'single',
                'path': '',
                'image_size': 640
            },
            'training': {
                'epochs': 300,
                'batch_size': 16,
                'device': '0'
            },
            'model': {
                'config': 'cfg/training/yolov7.yaml',
                'weights': ''
            }
        }
        
        return default_config
    
    def get_training_config(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """UI 설정을 YOLOv7 훈련 설정으로 변환 - 하이퍼파라미터 분리"""
        
        # 모델 설정 파일 경로 처리
        model_config_input = ui_config.get('model_config', 'cfg/training/yolov7.yaml')
        model_config_path = self.resolve_model_config_path(model_config_input)
        
        # 가중치 파일 경로 처리
        weights_path = ui_config.get('weights_path', '')
        if weights_path:
            weights_path = self.resolve_weights_path(weights_path)
        
        # 하이퍼파라미터 파일 경로 처리
        hyperparams_file = ui_config.get('hyperparams_file', '')
        if hyperparams_file:
            hyperparams_file = self.resolve_hyperparams_path(hyperparams_file)
        
        config = {
            # 기본 훈련 설정
            'dataset_path': ui_config.get('dataset_path', ''),
            'model_config': model_config_path,
            'weights_path': weights_path,
            'hyperparams_file': hyperparams_file,  # 🔥 하이퍼파라미터 파일 경로
            
            # 훈련 파라미터
            'epochs': ui_config.get('epochs', 300),
            'batch_size': ui_config.get('batch_size', 16),
            'image_size': ui_config.get('image_size', 640),
            'device': ui_config.get('device', '0'),
            'workers': ui_config.get('workers', 8),
            'experiment_name': ui_config.get('experiment_name', 'exp'),
            
            # 훈련 옵션들 (하이퍼파라미터와 별개)
            'cache_images': ui_config.get('cache_images', False),
            'multi_scale': ui_config.get('multi_scale', False),
            'image_weights': ui_config.get('image_weights', False),
            'rect': ui_config.get('rect', False),
            'adam': ui_config.get('adam', False),
            'sync_bn': ui_config.get('sync_bn', False),
            'single_cls': ui_config.get('single_cls', False),
            
            # 추가 옵션들
            'notest': ui_config.get('notest', False),
            'evolve': ui_config.get('evolve', False),
            'resume': ui_config.get('resume', ''),
        }
        
        return config

    def resolve_hyperparams_path(self, hyperparams_input: str) -> str:
        """하이퍼파라미터 파일 경로 해결"""
        
        if not hyperparams_input:
            return ''
        
        hyperparams_path = Path(hyperparams_input)
        
        # 절대 경로이고 존재하면 그대로 사용
        if hyperparams_path.is_absolute() and hyperparams_path.exists():
            return str(hyperparams_path)
        
        # 상대 경로인 경우 YOLOv7 디렉토리에서 검색
        if not hyperparams_path.is_absolute():
            absolute_path = self.yolo_dir / hyperparams_input
            if absolute_path.exists():
                safe_print(f"✅ 하이퍼파라미터 파일 발견: {absolute_path}")
                return str(absolute_path)
        
        # 파일을 찾을 수 없는 경우 자동 검색
        safe_print(f"⚠️ 하이퍼파라미터 파일을 찾을 수 없음: {hyperparams_input}")
        safe_print("🔍 자동으로 기본 파일을 검색 중...")
        
        # 검색할 경로들
        search_paths = [
            self.yolo_dir / 'data',
            self.yolo_dir / 'cfg', 
            self.yolo_dir,
        ]
        
        # 검색할 파일명들 (우선순위 순)
        search_files = [
            hyperparams_path.name,  # 원본 파일명
            'hyp.scratch.p5.yaml',  # 기본 P5
            'hyp.scratch.p6.yaml',  # P6 
            'hyp.finetune.yaml'     # Fine-tuning
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for filename in search_files:
                    candidate = search_path / filename
                    if candidate.exists():
                        safe_print(f"✅ 대체 하이퍼파라미터 파일 발견: {candidate}")
                        return str(candidate)
        
        # 모든 시도 실패 시 빈 문자열 반환 (기본값 사용)
        safe_print(f"❌ 하이퍼파라미터 파일을 찾을 수 없습니다. YOLOv7 기본값 사용")
        return ''
    def resolve_model_config_path(self, config_input: str) -> str:
        """🔥 모델 설정 파일 경로 해결 (새로 추가된 메서드)"""
        
        # 1. 절대 경로인 경우 그대로 사용
        config_path = Path(config_input)
        if config_path.is_absolute() and config_path.exists():
            safe_print(f"✅ 절대 경로 모델 설정 사용: {config_path}")
            return str(config_path)
        
        # 2. 상대 경로인 경우 YOLOv7 디렉토리 기준으로 변환
        if not config_path.is_absolute():
            absolute_path = self.yolo_dir / config_input
            if absolute_path.exists():
                safe_print(f"✅ 상대 경로 해결: {absolute_path}")
                return str(absolute_path)
        
        # 3. 파일을 찾을 수 없는 경우 자동 검색
        safe_print(f"⚠️ 설정 파일을 찾을 수 없음: {config_input}")
        safe_print("🔍 자동으로 대체 파일을 검색 중...")
        
        # 검색할 경로들
        search_paths = [
            self.yolo_dir / 'cfg' / 'training',
            self.yolo_dir / 'cfg',
            self.yolo_dir,
        ]
        
        # 검색할 파일명들 (우선순위 순)
        search_files = [
            config_path.name,  # 원본 파일명
            'yolov7.yaml',     # 기본 YOLOv7
            'yolov7x.yaml',    # YOLOv7-X
            'yolov7-tiny.yaml' # YOLOv7-Tiny
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for filename in search_files:
                    candidate = search_path / filename
                    if candidate.exists():
                        safe_print(f"✅ 대체 모델 설정 발견: {candidate}")
                        return str(candidate)
        
        # 4. 모든 시도 실패 시 원본 경로 반환 (오류 발생 예상)
        safe_print(f"❌ 모델 설정 파일을 찾을 수 없습니다: {config_input}")
        return str(self.yolo_dir / config_input)  # 최소한 절대 경로로 변환
    
    def resolve_weights_path(self, weights_input: str) -> str:
        """🔥 가중치 파일 경로 해결 (새로 추가된 메서드)"""
        
        if not weights_input:
            return ''
        
        weights_path = Path(weights_input)
        
        # 절대 경로이고 존재하면 그대로 사용
        if weights_path.is_absolute() and weights_path.exists():
            return str(weights_path)
        
        # 상대 경로인 경우 YOLOv7 디렉토리에서 검색
        if not weights_path.is_absolute():
            absolute_path = self.yolo_dir / weights_input
            if absolute_path.exists():
                return str(absolute_path)
        
        # YOLOv7 루트 디렉토리에서 파일명으로 검색
        filename = weights_path.name
        candidate = self.yolo_dir / filename
        if candidate.exists():
            safe_print(f"✅ 가중치 파일 발견: {candidate}")
            return str(candidate)
        
        # weights 폴더에서 검색
        weights_folder = self.yolo_dir / 'weights'
        if weights_folder.exists():
            candidate = weights_folder / filename
            if candidate.exists():
                safe_print(f"✅ weights 폴더에서 가중치 발견: {candidate}")
                return str(candidate)
        
        safe_print(f"⚠️ 가중치 파일을 찾을 수 없음: {weights_input}")
        return str(weights_path)  # 원본 경로 반환
    
    def validate_config(self, config: Dict[str, Any]):
        """설정 유효성 검사 (호환성 버전)"""
        
        # 필수 항목 체크
        required_fields = ['dataset_path', 'model_config', 'epochs', 'batch_size']
        for field in required_fields:
            if not config.get(field):
                return False, f"필수 항목 누락: {field}"
        
        # 🔥 파일 존재 여부 추가 검증
        model_config_path = Path(config['model_config'])
        if not model_config_path.exists():
            return False, f"모델 설정 파일을 찾을 수 없음: {model_config_path}"
        
        dataset_path = Path(config['dataset_path'])
        if config['dataset_path'] and not dataset_path.exists():
            return False, f"데이터셋 파일을 찾을 수 없음: {dataset_path}"
        
        return True, "설정이 유효합니다"
    
    def get_available_model_configs(self):
        """🔥 사용 가능한 모델 설정 파일 목록 반환 (새로 추가된 메서드)"""
        available_configs = []
        
        search_paths = [
            self.yolo_dir / 'cfg' / 'training',
            self.yolo_dir / 'cfg',
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for yaml_file in search_path.glob('*.yaml'):
                    if 'yolov7' in yaml_file.name.lower():
                        available_configs.append({
                            'name': yaml_file.name,
                            'path': str(yaml_file),
                            'relative_path': str(yaml_file.relative_to(self.yolo_dir))
                        })
        
        return available_configs


# 🔥 테스트 함수 개선
if __name__ == "__main__":
    safe_print("🧪 ConfigManager 향상된 테스트...")
    
    try:
        config_manager = ConfigManager()
        
        # 사용 가능한 모델 설정 파일 확인
        available_configs = config_manager.get_available_model_configs()
        safe_print(f"📁 사용 가능한 모델 설정: {len(available_configs)}개")
        for config in available_configs:
            safe_print(f"   - {config['name']}")
        
        # 샘플 UI 설정
        sample_ui_config = {
            'dataset_path': 'path/to/dataset.yaml',
            'model_config': 'cfg/training/yolov7.yaml',  # 상대 경로 테스트
            'weights_path': 'yolov7.pt',                  # 가중치 파일 테스트
            'epochs': 100,
            'batch_size': 8,
            'image_size': 640,
            'device': '0',
            'experiment_name': 'test_exp'
        }
        
        yolo_config = config_manager.get_training_config(sample_ui_config)
        safe_print(f"✅ 설정 변환 성공: {len(yolo_config)} 항목")
        safe_print(f"📂 해결된 모델 설정 경로: {yolo_config['model_config']}")
        safe_print(f"⚖️ 해결된 가중치 경로: {yolo_config['weights_path']}")
        
        # 설정 유효성 검사
        is_valid, message = config_manager.validate_config(yolo_config)
        safe_print(f"🔍 설정 검증: {message}")
        
    except Exception as e:
        safe_print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()