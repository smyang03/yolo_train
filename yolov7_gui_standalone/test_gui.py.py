import os
import sys
import multiprocessing

def optimize_for_exe():
    """EXE 환경 최적화"""
    
    # 멀티프로세싱 설정
    if sys.platform.startswith('win'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except:
            pass
    
    # 환경 변수 설정
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    
    # PyTorch 설정
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

def get_system_info():
    """시스템 정보 수집"""
    info = {
        'platform': sys.platform,
        'python_version': sys.version.split()[0],
        'exe_mode': hasattr(sys, '_MEIPASS'),
        'cuda_available': False,
        'cuda_device_count': 0
    }
    
    try:
        import torch
        info['pytorch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['pytorch_version'] = 'Not installed'
    
    try:
        import cv2
        info['opencv_version'] = cv2.__version__
    except ImportError:
        info['opencv_version'] = 'Not installed'
    
    return info


# GUI 테스트 스크립트 - test_gui.py

import sys
from pathlib import Path

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_gui():
    """GUI 테스트"""
    
    print("🧪 YOLOv7 GUI 테스트 시작...")
    
    try:
        from app import YOLOv7App
        
        app = YOLOv7App()
        print("✅ 애플리케이션 초기화 성공!")
        
        # GUI 실행
        app.run()
        
    except Exception as e:
        print(f"❌ GUI 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 현재 디렉토리 확인
    if Path.cwd().name != "yolov7_gui_standalone":
        print("❌ yolov7_gui_standalone 폴더에서 실행하세요!")
        print(f"현재 위치: {Path.cwd()}")
        exit(1)
    
    test_gui()