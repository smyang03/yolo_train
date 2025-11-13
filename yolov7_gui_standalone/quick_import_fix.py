# quick_import_fix.py - 임포트 문제 빠른 수정

from pathlib import Path

def fix_import_issues():
    """임포트 문제 수정"""
    
    print("🔧 임포트 문제 수정 중...")
    
    # src/app.py 수정
    app_content = '''"""
YOLOv7 Training GUI - Main Application
"""

import sys
import os
import tkinter as tk
from pathlib import Path

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class YOLOv7App:
    def __init__(self):
        print("🎯 YOLOv7 GUI 애플리케이션 초기화 중...")
        self.setup_paths()
        self.setup_environment()
        self.init_components()
    
    def setup_paths(self):
        self.app_dir = Path(__file__).parent.parent
        self.resources_dir = Path(get_resource_path("resources"))
        self.output_dir = self.app_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 앱 디렉토리: {self.app_dir}")
    
    def setup_environment(self):
        try:
            from utils.system_utils import get_system_info
            self.system_info = get_system_info()
            print("✅ 시스템 정보 로드 완료")
        except ImportError:
            self.system_info = {"platform": sys.platform}
            print("⚠️ 시스템 유틸리티 로드 실패")
    
    def init_components(self):
        try:
            from core.yolo_trainer import YOLOv7Trainer
            from core.config_manager import ConfigManager
            from core.model_manager import ModelManager
            
            self.trainer = YOLOv7Trainer()
            self.config_manager = ConfigManager()
            self.model_manager = ModelManager()
            print("✅ 핵심 컴포넌트 로드 완료")
        except Exception as e:
            print(f"❌ 컴포넌트 로드 실패: {e}")
            raise
    
    def run(self):
        try:
            print("🚀 GUI 시작 중...")
            root = tk.Tk()
            
            # 간단한 테스트 창 표시
            root.title("YOLOv7 GUI - 연결 테스트")
            root.geometry("600x400")
            
            # 환영 메시지
            import tkinter.ttk as ttk
            
            main_frame = ttk.Frame(root, padding="20")
            main_frame.pack(fill='both', expand=True)
            
            title_label = ttk.Label(main_frame, text="🚀 YOLOv7 Training GUI", 
                                   font=('Arial', 18, 'bold'))
            title_label.pack(pady=20)
            
            status_label = ttk.Label(main_frame, text="✅ 연결 성공! 모든 모듈이 정상적으로 로드되었습니다.", 
                                    font=('Arial', 12))
            status_label.pack(pady=10)
            
            # 시스템 정보 표시
            info_frame = ttk.LabelFrame(main_frame, text="시스템 정보", padding="10")
            info_frame.pack(fill='x', pady=20)
            
            for key, value in self.system_info.items():
                info_label = ttk.Label(info_frame, text=f"{key}: {value}")
                info_label.pack(anchor='w')
            
            # 버튼들
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=20)
            
            test_btn = ttk.Button(button_frame, text="연결 테스트", command=self.test_connection)
            test_btn.pack(side='left', padx=5)
            
            close_btn = ttk.Button(button_frame, text="닫기", command=root.quit)
            close_btn.pack(side='left', padx=5)
            
            print("✅ GUI 시작 완료")
            root.mainloop()
            
        except Exception as e:
            self.handle_error(e)
    
    def test_connection(self):
        """연결 테스트"""
        try:
            # 간단한 설정 테스트
            test_config = {
                'dataset_path': 'test.yaml',
                'model_config': 'cfg/training/yolov7.yaml',
                'epochs': 1,
                'batch_size': 1,
                'image_size': 640,
                'device': 'cpu',
                'experiment_name': 'test'
            }
            
            yolo_config = self.config_manager.get_training_config(test_config)
            cmd = self.trainer.build_command(yolo_config)
            
            print("🧪 연결 테스트 성공!")
            print(f"🔧 생성된 명령어: {' '.join(str(x) for x in cmd[:5])}...")
            
            import tkinter.messagebox as msgbox
            msgbox.showinfo("테스트 성공", "YOLOv7 연결이 정상적으로 작동합니다!")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import tkinter.messagebox as msgbox
            msgbox.showerror("테스트 실패", f"연결 테스트 실패: {e}")
    
    def handle_error(self, error):
        print(f"❌ 오류: {error}")
        import traceback
        traceback.print_exc()
'''
    
    with open("src/app.py", 'w', encoding='utf-8') as f:
        f.write(app_content)
    
    print("✅ src/app.py 수정 완료!")
    
    # src/utils/system_utils.py 확인/생성
    utils_file = Path("src/utils/system_utils.py")
    if not utils_file.exists():
        utils_content = '''"""
시스템 유틸리티
"""

import os
import sys

def get_system_info():
    info = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0]
    }
    
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        info["pytorch_version"] = "Not installed"
    
    try:
        import cv2
        info["opencv_version"] = cv2.__version__
    except ImportError:
        info["opencv_version"] = "Not installed"
    
    return info

def optimize_for_exe():
    pass
'''
        
        with open(utils_file, 'w', encoding='utf-8') as f:
            f.write(utils_content)
        
        print("✅ src/utils/system_utils.py 생성 완료!")
    
    print("🎉 임포트 문제 수정 완료!")

if __name__ == "__main__":
    if Path.cwd().name != "yolov7_gui_standalone":
        print("❌ yolov7_gui_standalone 폴더에서 실행하세요!")
        exit(1)
    
    fix_import_issues()