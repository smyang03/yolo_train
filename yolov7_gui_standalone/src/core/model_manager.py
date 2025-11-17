"""
from utils import safe_print
모델 관리 모듈 (Python 3.8+ 호환)
"""

import sys
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import os

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == 'win32':
    try:
        if sys.version_info >= (3, 7):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

class ModelManager:
    """훈련된 모델 관리 클래스"""

    def __init__(self):
        # PyInstaller 환경 감지
        if getattr(sys, 'frozen', False):
            self.app_dir = Path(sys.executable).parent
        else:
            self.app_dir = Path(__file__).parent.parent.parent

        self.output_dir = self.app_dir / "outputs"
        self.models_dir = self.app_dir / "saved_models"

        self.models_dir.mkdir(exist_ok=True, parents=True)
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
                safe_print(f"모델 파일 정보 읽기 실패: {e}")
    
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
    safe_print("🧪 ModelManager 테스트...")
    
    try:
        model_manager = ModelManager()
        summary = model_manager.get_model_summary()
        safe_print(f"✅ 모델 관리자 초기화 성공! 총 모델: {summary['total_models']}개")
        
    except Exception as e:
        safe_print(f"❌ 테스트 실패: {e}")
