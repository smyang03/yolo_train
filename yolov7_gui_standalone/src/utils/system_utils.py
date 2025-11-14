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

def get_available_devices():
    """
    사용 가능한 GPU 디바이스 목록을 자동으로 감지하여 반환

    Returns:
        tuple: (device_list, default_device)
            - device_list: 사용 가능한 디바이스 문자열 리스트
            - default_device: 기본값으로 사용할 디바이스

    Examples:
        GPU 4개: (["0", "1", "2", "3", "0,1", "0,1,2,3", "cpu"], "0,1,2,3")
        GPU 2개: (["0", "1", "0,1", "cpu"], "0,1")
        GPU 1개: (["0", "cpu"], "0")
        GPU 없음: (["cpu"], "cpu")
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return (["cpu"], "cpu")

        gpu_count = torch.cuda.device_count()

        if gpu_count == 0:
            return (["cpu"], "cpu")

        # 개별 GPU 옵션 생성
        device_list = [str(i) for i in range(gpu_count)]

        # 멀티 GPU 조합 생성
        if gpu_count > 1:
            # 2개 조합
            if gpu_count >= 2:
                device_list.append("0,1")

            # 4개 조합
            if gpu_count >= 4:
                device_list.append("0,1,2,3")

            # 전체 조합 (2개나 4개가 아닌 경우)
            if gpu_count not in [2, 4]:
                all_gpus = ",".join(str(i) for i in range(gpu_count))
                if all_gpus not in device_list:
                    device_list.append(all_gpus)

        # CPU 옵션 추가
        device_list.append("cpu")

        # 기본값: 모든 GPU 사용
        default_device = ",".join(str(i) for i in range(gpu_count))

        return (device_list, default_device)

    except ImportError:
        # PyTorch가 설치되지 않은 경우
        return (["cpu"], "cpu")
    except Exception as e:
        # 기타 오류 발생 시 안전한 기본값 반환
        print(f"⚠️ GPU 감지 오류: {e}")
        return (["0", "cpu"], "0")

def get_optimal_workers():
    """
    플랫폼에 맞는 최적의 DataLoader workers 수를 반환

    Windows는 multiprocessing 메모리 문제로 인해 workers를 제한
    ⚠️ workers=0은 persistent_workers 오류를 일으킬 수 있어 최소 1로 설정

    Returns:
        int: 권장 workers 수

    Examples:
        Windows: 1 (persistent_workers 호환, 안정적)
        Linux: 4~8 (멀티프로세싱 활용)
    """
    import platform

    # Windows 감지
    is_windows = sys.platform.startswith('win') or platform.system() == 'Windows'

    if is_windows:
        # Windows: workers=1 권장 (0은 persistent_workers 오류 발생)
        # workers=1이면 메모리 문제 없으면서도 YOLOv7 호환
        return 1
    else:
        # Linux/Unix: CPU 코어 수 기반으로 설정
        try:
            cpu_count = multiprocessing.cpu_count()
            # 최대 8개까지만
            return min(cpu_count // 2, 8)
        except:
            return 4

def validate_workers(workers, show_warning=False):
    """
    설정된 workers 수가 적절한지 검증

    Args:
        workers: 설정하려는 workers 수
        show_warning: 경고 메시지 출력 여부

    Returns:
        tuple: (is_safe, warning_message)
    """
    import platform

    is_windows = sys.platform.startswith('win') or platform.system() == 'Windows'

    # ⚠️ workers=0은 persistent_workers 오류 발생!
    if workers == 0:
        warning_msg = (
            f"⚠️ Workers=0은 YOLOv7에서 오류를 일으킵니다!\n\n"
            f"오류 메시지:\n"
            f"  'ValueError: persistent_workers option needs num_workers > 0'\n\n"
            f"Workers를 1로 변경하시겠습니까? (권장)"
        )
        return (False, warning_msg)

    if is_windows and workers > 2:
        warning_msg = (
            f"⚠️ Windows에서 Workers={workers}는 메모리 문제를 일으킬 수 있습니다.\n\n"
            f"권장값:\n"
            f"  • workers=1 (가장 안정적, YOLOv7 호환)\n"
            f"  • workers=2 (약간 더 빠름)\n\n"
            f"현재 값({workers})을 사용하면 훈련 중 다음 오류가 발생할 수 있습니다:\n"
            f"  'OSError: [WinError 1455] 페이징 파일이 너무 작습니다'\n\n"
            f"Workers를 1로 변경하시겠습니까?"
        )
        return (False, warning_msg)

    return (True, "")

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