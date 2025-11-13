# fix_type_hints.py - 타입 힌트 호환성 문제 수정

from pathlib import Path
import re

def fix_type_hints():
    """타입 힌트 호환성 문제 수정"""
    
    print("🔧 타입 힌트 문제 수정 중...")
    
    files_to_fix = [
        "src/core/config_manager.py",
        "src/core/model_manager.py",
        "src/core/yolo_trainer.py"
    ]
    
    for file_path in files_to_fix:
        fix_file_type_hints(Path(file_path))
    
    print("✅ 모든 타입 힌트 수정 완료!")

def fix_file_type_hints(file_path: Path):
    """개별 파일의 타입 힌트 수정"""
    
    if not file_path.exists():
        print(f"⚠️ 파일 없음: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 타입 힌트 수정
        # tuple[bool, str] → Tuple[bool, str]
        content = re.sub(r'\btuple\[', 'Tuple[', content)
        
        # list[...] → List[...]
        content = re.sub(r'\blist\[', 'List[', content)
        
        # dict[...] → Dict[...]
        content = re.sub(r'\bdict\[', 'Dict[', content)
        
        # typing 임포트 확인 및 추가
        if 'from typing import' in content:
            # 이미 typing 임포트가 있는 경우, Tuple 추가
            if 'Tuple' not in content and 'tuple[' in content:
                content = content.replace(
                    'from typing import',
                    'from typing import Tuple,'
                )
        else:
            # typing 임포트가 없는 경우 추가
            if 'tuple[' in content or 'list[' in content or 'dict[' in content:
                lines = content.split('\n')
                # import 섹션 찾기
                import_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_idx = i
                
                # typing 임포트 추가
                lines.insert(import_idx + 1, 'from typing import Dict, Any, List, Tuple')
                content = '\n'.join(lines)
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 수정 완료: {file_path}")
        
    except Exception as e:
        print(f"❌ 수정 실패: {file_path} - {e}")

def create_compatible_config_manager():
    """호환성 있는 config_manager.py 생성"""
    
    compatible_code = '''"""
설정 관리 모듈 (Python 3.8+ 호환)
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.app_dir / "resources" / "configs"
        self.default_config_path = self.config_dir / "default.yaml"
        
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
        """UI 설정을 YOLOv7 훈련 설정으로 변환"""
        
        app_dir = Path(__file__).parent.parent.parent
        yolo_dir = app_dir.parent / "yolov7"
        
        config = {
            'dataset_path': ui_config.get('dataset_path', ''),
            'model_config': yolo_dir / ui_config.get('model_config', 'cfg/training/yolov7.yaml'),
            'epochs': ui_config.get('epochs', 300),
            'batch_size': ui_config.get('batch_size', 16),
            'image_size': ui_config.get('image_size', 640),
            'device': ui_config.get('device', '0'),
            'experiment_name': ui_config.get('experiment_name', 'exp')
        }
        
        return config
    
    def validate_config(self, config: Dict[str, Any]):
        """설정 유효성 검사 (호환성 버전)"""
        
        # 필수 항목 체크
        required_fields = ['dataset_path', 'model_config', 'epochs', 'batch_size']
        for field in required_fields:
            if not config.get(field):
                return False, f"필수 항목 누락: {field}"
        
        return True, "설정이 유효합니다"


# 테스트 함수
if __name__ == "__main__":
    print("🧪 ConfigManager 테스트...")
    
    try:
        config_manager = ConfigManager()
        
        # 샘플 UI 설정
        sample_ui_config = {
            'dataset_path': 'path/to/dataset.yaml',
            'model_config': 'cfg/training/yolov7.yaml',
            'epochs': 100,
            'batch_size': 8,
            'image_size': 640,
            'device': '0',
            'experiment_name': 'test_exp'
        }
        
        yolo_config = config_manager.get_training_config(sample_ui_config)
        print(f"✅ 설정 변환 성공: {len(yolo_config)} 항목")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
'''
    
    with open("src/core/config_manager.py", 'w', encoding='utf-8') as f:
        f.write(compatible_code)
    
    print("✅ 호환성 있는 config_manager.py 생성 완료!")

def create_compatible_model_manager():
    """호환성 있는 model_manager.py 생성"""
    
    compatible_code = '''"""
모델 관리 모듈 (Python 3.8+ 호환)
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import os

class ModelManager:
    """훈련된 모델 관리 클래스"""
    
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent.parent
        self.output_dir = self.app_dir / "outputs"
        self.models_dir = self.app_dir / "saved_models"
        
        self.models_dir.mkdir(exist_ok=True)
        self.saved_models = []
        self.load_saved_models()
    
    def load_saved_models(self):
        """저장된 모델 목록 로드"""
        self.saved_models = []
        
        if self.output_dir.exists():
            for exp_dir in self.output_dir.iterdir():
                if exp_dir.is_dir():
                    weights_dir = exp_dir / "weights"
                    if weights_dir.exists():
                        self._scan_weights_directory(weights_dir, exp_dir.name)
    
    def _scan_weights_directory(self, weights_dir: Path, experiment_name: str):
        """weights 디렉토리 스캔"""
        for weight_file in weights_dir.glob("*.pt"):
            try:
                stat = weight_file.stat()
                
                model_info = {
                    'filepath': weight_file,
                    'filename': weight_file.name,
                    'experiment': experiment_name,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'created_time': datetime.fromtimestamp(stat.st_mtime),
                    'type': self._determine_model_type(weight_file.name)
                }
                
                self.saved_models.append(model_info)
                
            except Exception as e:
                print(f"모델 파일 정보 읽기 실패: {e}")
    
    def _determine_model_type(self, filename: str) -> str:
        """파일명으로 모델 타입 결정"""
        filename_lower = filename.lower()
        
        if 'best' in filename_lower:
            return 'best'
        elif 'last' in filename_lower:
            return 'last'
        else:
            return 'checkpoint'
    
    def get_model_summary(self) -> Dict[str, Any]:
        """모델 요약 정보 반환"""
        total_models = len(self.saved_models)
        total_size_mb = sum(m['size_mb'] for m in self.saved_models)
        
        return {
            'total_models': total_models,
            'total_size_mb': round(total_size_mb, 2),
            'latest_model': max(self.saved_models, key=lambda x: x['created_time']) if self.saved_models else None
        }


# 테스트 함수
if __name__ == "__main__":
    print("🧪 ModelManager 테스트...")
    
    try:
        model_manager = ModelManager()
        summary = model_manager.get_model_summary()
        print(f"✅ 모델 관리자 초기화 성공! 총 모델: {summary['total_models']}개")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
'''
    
    with open("src/core/model_manager.py", 'w', encoding='utf-8') as f:
        f.write(compatible_code)
    
    print("✅ 호환성 있는 model_manager.py 생성 완료!")

if __name__ == "__main__":
    if Path.cwd().name != "yolov7_gui_standalone":
        print("❌ yolov7_gui_standalone 폴더에서 실행하세요!")
        exit(1)
    
    # 호환성 있는 파일들로 교체
    create_compatible_config_manager()
    create_compatible_model_manager()
    
    print("\n🎉 타입 힌트 호환성 문제 해결 완료!")
    print("📋 다음 단계: python test_gui.py")