"""
YOLOv7 Training GUI - Main Entry Point
메인 실행 파일
"""

import sys
import os
import traceback
from pathlib import Path
import io

# Windows 콘솔 UTF-8 인코딩 설정 (이모지 및 한글 출력 지원)
if sys.platform == 'win32':
    try:
        # Python 3.7+에서는 UTF-8 모드 활성화
        if sys.version_info >= (3, 7):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 경로 설정
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

def get_resource_path(relative_path):
    """EXE에서 리소스 파일 경로 찾기"""
    try:
        # PyInstaller로 빌드된 경우
        base_path = sys._MEIPASS
    except Exception:
        # 개발 환경
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def check_requirements():
    """필수 패키지 확인"""
    required_packages = ['torch', 'torchvision', 'cv2', 'numpy', 'matplotlib', 'yaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 다음 패키지들이 설치되지 않았습니다:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 설치 명령어:")
        print("pip install torch torchvision opencv-python numpy matplotlib PyYAML")
        return False
    
    return True

def main():
    """메인 실행 함수"""

    print("🚀 YOLOv7 Training GUI 시작...")
    print("=" * 50)

    try:
        # 필수 패키지 확인
        if not check_requirements():
            input("\n패키지를 설치한 후 Enter를 누르세요...")
            return

        print("✅ 모든 필수 패키지가 설치되어 있습니다.")

        # GUI 애플리케이션 시작
        from app import YOLOv7App

        print("🎯 애플리케이션 초기화 중...")
        app = YOLOv7App()

        print("🎨 Professional GUI 시작 중...")
        app.run()

    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 종료되었습니다.")

    except ImportError as e:
        print(f"❌ 모듈 임포트 오류: {e}")
        print(f"상세 정보: {traceback.format_exc()}")
        print("\n🔧 해결 방법:")
        print("1. 필요한 패키지가 설치되었는지 확인")
        print("2. 가상환경이 활성화되었는지 확인")
        print("3. Python 경로가 올바른지 확인")
        input("\nEnter를 눌러 종료...")

    except Exception as e:
        # 에러 로깅 - 자세한 정보 출력
        error_log = current_dir / "error.log"
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}\n\nSystem Info:\n"
        error_msg += f"Python: {sys.version}\n"
        error_msg += f"Platform: {sys.platform}\n"
        error_msg += f"Executable: {sys.executable}\n"
        error_msg += f"Current Dir: {current_dir}\n"

        try:
            with open(error_log, "w", encoding='utf-8') as f:
                f.write(error_msg)
        except:
            pass

        print(f"❌ 애플리케이션 오류: {str(e)}")
        print(f"\n상세 에러 정보:")
        print(traceback.format_exc())
        print(f"\n📝 자세한 오류 정보가 {error_log}에 저장되었습니다.")

        # 사용자에게 에러 알림
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "오류 발생",
                f"애플리케이션 실행 중 오류가 발생했습니다.\n\n{str(e)}\n\n자세한 내용은 error.log 파일을 확인하세요."
            )
        except:
            print("GUI 오류 알림 표시 실패")

        input("\nEnter를 눌러 종료...")
        sys.exit(1)

if __name__ == "__main__":
    main()
