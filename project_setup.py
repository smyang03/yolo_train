#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 GUI 프로젝트 구조 자동 생성 스크립트
폴더와 빈 파일들만 생성합니다.
"""

import os
from pathlib import Path

def create_project_structure():
    """YOLOv7 GUI 프로젝트 구조 생성"""
    
    print("🚀 YOLOv7 GUI 프로젝트 구조 생성 시작...")
    
    # 기본 프로젝트 구조 정의
    project_structure = [
        # 루트 파일들
        "yolov7_gui_standalone/main.py",
        "yolov7_gui_standalone/requirements.txt",
        "yolov7_gui_standalone/build_exe.py",
        "yolov7_gui_standalone/setup.cfg",
        "yolov7_gui_standalone/README.md",
        "yolov7_gui_standalone/.gitignore",
        
        # src 폴더
        "yolov7_gui_standalone/src/__init__.py",
        "yolov7_gui_standalone/src/app.py",
        
        # src/ui 폴더
        "yolov7_gui_standalone/src/ui/__init__.py",
        "yolov7_gui_standalone/src/ui/main_window.py",
        "yolov7_gui_standalone/src/ui/components.py",
        "yolov7_gui_standalone/src/ui/styles.py",
        
        # src/core 폴더
        "yolov7_gui_standalone/src/core/__init__.py",
        "yolov7_gui_standalone/src/core/yolo_trainer.py",
        "yolov7_gui_standalone/src/core/log_parser.py",
        "yolov7_gui_standalone/src/core/config_manager.py",
        "yolov7_gui_standalone/src/core/model_manager.py",
        
        # src/utils 폴더
        "yolov7_gui_standalone/src/utils/__init__.py",
        "yolov7_gui_standalone/src/utils/file_utils.py",
        "yolov7_gui_standalone/src/utils/system_utils.py",
        "yolov7_gui_standalone/src/utils/validation.py",
        
        # resources 폴더
        "yolov7_gui_standalone/resources/icons/.gitkeep",
        "yolov7_gui_standalone/resources/configs/default.yaml",
        "yolov7_gui_standalone/resources/configs/model_configs/.gitkeep",
        "yolov7_gui_standalone/resources/templates/.gitkeep",
        
        # yolov7_embedded 폴더
        "yolov7_gui_standalone/yolov7_embedded/__init__.py",
        "yolov7_gui_standalone/yolov7_embedded/train_core.py",
        "yolov7_gui_standalone/yolov7_embedded/models/__init__.py",
        "yolov7_gui_standalone/yolov7_embedded/models/yolo.py",
        "yolov7_gui_standalone/yolov7_embedded/models/common.py",
        "yolov7_gui_standalone/yolov7_embedded/utils/__init__.py",
        "yolov7_gui_standalone/yolov7_embedded/utils/general.py",
        "yolov7_gui_standalone/yolov7_embedded/utils/torch_utils.py",
        "yolov7_gui_standalone/yolov7_embedded/utils/datasets.py",
        "yolov7_gui_standalone/yolov7_embedded/cfg/training/.gitkeep",
        
        # build 폴더
        "yolov7_gui_standalone/build/spec_files/.gitkeep",
        "yolov7_gui_standalone/build/dist/.gitkeep",
        
        # tests 폴더
        "yolov7_gui_standalone/tests/__init__.py",
        "yolov7_gui_standalone/tests/test_ui.py",
        "yolov7_gui_standalone/tests/test_trainer.py",
        
        # docs 폴더
        "yolov7_gui_standalone/docs/user_manual.md",
        "yolov7_gui_standalone/docs/developer_guide.md"
    ]
    
    # 폴더와 파일 생성
    create_files_and_folders(project_structure)
    
    # 기본 내용이 있어야 하는 파일들
    create_basic_files()
    
    print("✅ 프로젝트 구조 생성 완료!")
    print("📁 생성된 폴더: yolov7_gui_standalone/")
    print()
    print("📋 다음 단계:")
    print("1. cd yolov7_gui_standalone")
    print("2. python -m venv venv")
    print("3. venv\\Scripts\\activate  (Windows)")
    print("4. pip install torch torchvision opencv-python matplotlib tkinter")
    print("5. 각 파일에 실제 코드 작성")

def create_files_and_folders(file_paths):
    """파일 경로 리스트를 받아서 폴더와 빈 파일들 생성"""
    
    created_folders = set()
    
    for file_path in file_paths:
        path = Path(file_path)
        
        # 폴더 생성
        folder = path.parent
        if folder not in created_folders:
            folder.mkdir(parents=True, exist_ok=True)
            created_folders.add(folder)
            print(f"📁 폴더 생성: {folder}")
        
        # 빈 파일 생성 (이미 존재하지 않는 경우만)
        if not path.exists():
            if path.suffix == '.py':
                # Python 파일은 기본 주석 추가
                content = f'"""\n{path.name}\nTODO: 구현 필요\n"""\n\n# TODO: 코드 작성\n'
            elif path.name == '.gitkeep':
                # .gitkeep 파일은 비워둠
                content = ''
            elif path.suffix in ['.md', '.txt', '.yaml', '.yml']:
                # 문서 파일들은 기본 제목 추가
                content = f'# {path.stem}\n\nTODO: 내용 작성\n'
            else:
                # 기타 파일들은 비워둠
                content = ''
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"📄 파일 생성: {path}")

def create_basic_files():
    """기본적인 내용이 필요한 파일들 생성"""
    
    # requirements.txt
    requirements_content = """# YOLOv7 GUI 기본 요구사항
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
matplotlib>=3.3.0
numpy>=1.21.0
PyYAML>=5.4.0
tqdm>=4.60.0
Pillow>=8.3.0
pandas>=1.3.0
"""
    
    with open("yolov7_gui_standalone/requirements.txt", 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/
*.egg

# Virtual Environment
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Project specific
outputs/
temp/
logs/
*.log
*.pt
*.weights
runs/
"""
    
    with open("yolov7_gui_standalone/.gitignore", 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    # README.md
    readme_content = """# YOLOv7 Training GUI

YOLOv7 객체 탐지 모델 훈련을 위한 GUI 애플리케이션

## 설치 방법

1. 가상환경 생성
```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
```

2. 패키지 설치
```bash
pip install -r requirements.txt
```

3. 실행
```bash
python main.py
```

## 프로젝트 구조

- `src/` - 소스 코드
- `resources/` - 리소스 파일들
- `yolov7_embedded/` - YOLOv7 핵심 코드
- `tests/` - 테스트 코드
- `docs/` - 문서

## TODO

- [ ] YOLOv7 연결 모듈 구현
- [ ] GUI 인터페이스 구현  
- [ ] 훈련 프로세스 통합
- [ ] EXE 빌드 설정
"""
    
    with open("yolov7_gui_standalone/README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # main.py (기본 진입점)
    main_content = '''"""
YOLOv7 Training GUI - Main Entry Point
"""

import sys
from pathlib import Path

# src 폴더를 Python 경로에 추가
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def main():
    """메인 함수"""
    try:
        print("🚀 YOLOv7 GUI 시작...")
        
        # TODO: GUI 애플리케이션 시작
        # from app import YOLOv7App
        # app = YOLOv7App()
        # app.run()
        
        print("⚠️ GUI 구현 필요")
        print("현재는 프로젝트 구조만 생성된 상태입니다.")
        
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("yolov7_gui_standalone/main.py", 'w', encoding='utf-8') as f:
        f.write(main_content)
    
    print("📝 기본 파일들 내용 추가 완료")

if __name__ == "__main__":
    create_project_structure()