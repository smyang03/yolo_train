# src/utils/system_utils.py - 시스템 유틸리티

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