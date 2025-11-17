"""
YOLOv7 Training GUI - Main Application
Professional GUI 전용 앱 클래스
"""

import sys
import os
import tkinter as tk
from pathlib import Path
import io

# Windows 콘솔 UTF-8 인코딩 설정 (이모지 및 한글 출력 지원)
if sys.platform == 'win32':
    try:
        if sys.version_info >= (3, 7):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def get_resource_path(relative_path):
    """리소스 파일 경로 반환"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class YOLOv7App:
    """YOLOv7 Professional GUI 메인 애플리케이션"""
    
    def __init__(self):
        print("🎯 YOLOv7 GUI 애플리케이션 초기화 중...")
        self.setup_paths()
        self.setup_environment() 
        self.init_components()
    
    def setup_paths(self):
        """경로 설정"""
        # PyInstaller 환경 감지
        if getattr(sys, 'frozen', False):
            # PyInstaller로 빌드된 EXE 실행 중
            # sys.executable은 EXE 파일 경로
            self.app_dir = Path(sys.executable).parent
            print(f"🔧 PyInstaller 모드: EXE 경로 사용")
        else:
            # 일반 Python 스크립트 실행 중
            self.app_dir = Path(__file__).parent.parent
            print(f"🔧 개발 모드: 스크립트 경로 사용")

        self.resources_dir = Path(get_resource_path("resources"))
        self.output_dir = self.app_dir / "outputs"

        try:
            self.output_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            print(f"⚠️ outputs 디렉토리 생성 실패: {e}")
            # 실행 파일과 같은 위치에 생성 시도
            self.output_dir = self.app_dir / "outputs"
            self.output_dir.mkdir(exist_ok=True, parents=True)

        print(f"📁 앱 디렉토리: {self.app_dir}")
        print(f"📁 리소스 디렉토리: {self.resources_dir}")
        print(f"📁 출력 디렉토리: {self.output_dir}")
    
    def setup_environment(self):
        """환경 설정"""
        try:
            from utils.system_utils import get_system_info
            self.system_info = get_system_info()
            print("✅ 시스템 정보 로드 완료")
        except ImportError as e:
            self.system_info = {"platform": sys.platform}
            print(f"⚠️ 시스템 유틸리티 로드 실패 (기본 설정 사용): {e}")
        except Exception as e:
            self.system_info = {"platform": sys.platform}
            print(f"⚠️ 시스템 정보 로드 중 오류 (기본 설정 사용): {e}")

    def init_components(self):
        """핵심 컴포넌트 초기화"""
        try:
            print("📦 핵심 모듈 임포트 중...")
            from core.yolo_trainer import YOLOv7Trainer
            from core.config_manager import ConfigManager
            from core.model_manager import ModelManager

            print("🔧 YOLOv7Trainer 초기화 중...")
            self.trainer = YOLOv7Trainer()

            print("🔧 ConfigManager 초기화 중...")
            self.config_manager = ConfigManager()

            print("🔧 ModelManager 초기화 중...")
            self.model_manager = ModelManager()

            if self.trainer.yolo_original_dir:
                print(f"✅ YOLOv7 경로 확인: {self.trainer.yolo_original_dir}")
            else:
                print("⚠️ YOLOv7 경로를 찾을 수 없습니다. 환경 변수를 설정하거나 수동으로 지정해야 합니다.")

            print("✅ 핵심 컴포넌트 로드 완료")

        except ImportError as e:
            import traceback
            print(f"❌ 모듈 임포트 실패: {e}")
            print(f"상세 정보:\n{traceback.format_exc()}")
            raise
        except Exception as e:
            import traceback
            print(f"❌ 컴포넌트 로드 실패: {e}")
            print(f"상세 정보:\n{traceback.format_exc()}")
            raise
    
    def run(self):
        """Professional GUI 실행"""
        try:
            print("🎨 Professional GUI 시작 중...")

            # 메인 윈도우 생성
            root = tk.Tk()
            root.withdraw()  # 일시적으로 숨김

            # 종료 시 cleanup 호출 등록
            root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(root))

            # Professional Main Window 로드
            from ui.main_window import MainWindow

            self.main_window = MainWindow(
                root=root,
                trainer=self.trainer,
                config_manager=self.config_manager,
                model_manager=self.model_manager
            )

            print("🚀 GUI 시작 중...")
            self.main_window.show()
            print("✅ GUI 시작 완료")

            # 연결 테스트 자동 실행
            self.auto_test_connection()

            # 메인 루프 시작
            root.mainloop()

        except Exception as e:
            self.handle_error(e)
        finally:
            # 항상 cleanup 호출
            self.cleanup()

    def on_closing(self, root):
        """창 닫기 시 안전한 종료"""
        print("🛑 애플리케이션 종료 요청...")

        # 훈련 중인지 확인
        if self.trainer.is_training:
            import tkinter.messagebox as msgbox
            result = msgbox.askyesno(
                "훈련 진행 중",
                "훈련이 진행 중입니다. 정말로 종료하시겠습니까?"
            )
            if not result:
                return

            # 훈련 중지
            print("훈련 중지 중...")
            self.trainer.stop_training()

        # 리소스 정리
        self.cleanup()

        # 창 닫기
        root.quit()
        root.destroy()
    
    def auto_test_connection(self):
        """자동 연결 테스트"""
        try:
            test_config = {
                'dataset_path': 'test.yaml',
                'model_config': 'cfg/training/yolov7.yaml',
                'epochs': 1,
                'batch_size': 1,
                'image_size': 640,
                'device': 'cpu',
                'experiment_name': 'connection_test'
            }
            
            yolo_config = self.config_manager.get_training_config(test_config)
            cmd = self.trainer.build_command(yolo_config)
            
            print("🧪 연결 테스트 성공!")
            print(f"🔧 생성된 명령어: {' '.join(cmd[:3])}...")
            
        except Exception as e:
            print(f"⚠️ 자동 연결 테스트 실패: {e}")
    
    def handle_error(self, error):
        """오류 처리"""
        print(f"❌ 애플리케이션 오류: {error}")
        
        # GUI 오류 표시
        try:
            import tkinter.messagebox as msgbox
            msgbox.showerror("오류", f"애플리케이션 오류가 발생했습니다:\n{error}")
        except:
            pass
    
    def cleanup(self):
        """정리 작업 - 모든 리소스 해제"""
        print("🧹 애플리케이션 정리 중...")

        try:
            # Trainer 리소스 정리
            if hasattr(self, 'trainer') and self.trainer:
                self.trainer.cleanup()

            # Config Manager 정리 (필요시)
            if hasattr(self, 'config_manager') and self.config_manager:
                pass  # 필요한 정리 작업

            # Model Manager 정리 (필요시)
            if hasattr(self, 'model_manager') and self.model_manager:
                pass  # 필요한 정리 작업

            print("✅ 정리 완료")

        except Exception as e:
            print(f"⚠️ 정리 중 오류: {e}")

        finally:
            print("👋 애플리케이션 종료")
