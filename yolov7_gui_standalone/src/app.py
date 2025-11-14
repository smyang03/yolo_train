"""
YOLOv7 Training GUI - Main Application
Professional GUI 전용 앱 클래스
"""

import sys
import os
import tkinter as tk
from pathlib import Path

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
        self.app_dir = Path(__file__).parent.parent
        self.resources_dir = Path(get_resource_path("resources"))
        self.output_dir = self.app_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 앱 디렉토리: {self.app_dir}")
    
    def setup_environment(self):
        """환경 설정"""
        try:
            from utils.system_utils import get_system_info
            self.system_info = get_system_info()
            print("✅ 시스템 정보 로드 완료")
        except ImportError:
            self.system_info = {"platform": sys.platform}
            print("⚠️ 시스템 유틸리티 로드 실패 (기본 설정 사용)")
    
    def init_components(self):
        """핵심 컴포넌트 초기화"""
        try:
            from core.yolo_trainer import YOLOv7Trainer
            from core.config_manager import ConfigManager  
            from core.model_manager import ModelManager
            
            self.trainer = YOLOv7Trainer()
            self.config_manager = ConfigManager()
            self.model_manager = ModelManager()
            
            print(f"✅ YOLOv7 경로 확인: {self.trainer.yolo_original_dir}")
            print("✅ 핵심 컴포넌트 로드 완료")
            
        except Exception as e:
            print(f"❌ 컴포넌트 로드 실패: {e}")
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
