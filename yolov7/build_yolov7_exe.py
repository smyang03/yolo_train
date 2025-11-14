# 1. 먼저 PyInstaller 설치
# pip install pyinstaller

# 2. 필요한 모듈 임포트
import os
import sys
import PyInstaller.__main__
import shutil

# 3. YOLOv7 프로젝트 경로 설정
yolov7_path = r"D:\code\YOLOV7"
os.chdir(yolov7_path)

# 4. 메인 스크립트 파일명 (YOLOv7의 주요 실행 파일)
# YOLOv7의 메인 스크립트를 지정하세요 (예: detect.py, train.py 등)
main_script = "detect.py"  # 실제 메인 파일명으로 변경하세요

# 5. 추가 데이터 파일과 폴더 지정
# YOLOv7 프로젝트에서 필요한 모든 데이터 파일과 폴더를 지정합니다
data_files = [
    "models",           # 모델 정의 폴더
    "utils",            # 유틸리티 스크립트
    "data",             # 데이터 설정 파일
    "cfg",              # 설정 파일
    "weights/*.pt",     # 가중치 파일
    # 추가로 필요한 파일이나 폴더가 있다면 여기에 추가하세요
]

# 6. PyInstaller 명령어 구성
pyinstaller_args = [
    main_script,
    "--name=YOLOv7_App",
    "--onefile",                # 단일 실행 파일로 빌드
    "--console",                # 콘솔 창 표시 (디버깅 목적)
    # "--windowed",             # GUI 앱으로 빌드하려면 "--console" 대신 이 옵션 사용
    "--clean",                  # 임시 파일 정리
    f"--workpath={os.path.join(yolov7_path, 'build')}",
    f"--distpath={os.path.join(yolov7_path, 'dist')}",
    f"--specpath={yolov7_path}",
]

# 7. 모든 데이터 파일 추가
for file_path in data_files:
    full_path = os.path.join(yolov7_path, file_path)
    if os.path.exists(full_path):
        if os.path.isdir(full_path):
            pyinstaller_args.append(f"--add-data={full_path};{file_path}")
        else:
            # 파일이나 와일드카드 패턴인 경우 처리
            import glob
            for file in glob.glob(full_path):
                rel_path = os.path.relpath(os.path.dirname(file), yolov7_path)
                pyinstaller_args.append(f"--add-data={file};{rel_path}")
    else:
        print(f"경고: {full_path}가 존재하지 않습니다.")

# 8. 숨겨진 임포트 추가 (YOLOv7에서 동적으로 임포트되는 모듈)
# 프로젝트에 따라 필요한 모듈을 추가하세요
hidden_imports = [
    "torch",
    "torchvision", 
    "cv2",
    "numpy",
    "PIL",
    "yaml",
    "tqdm",
    "seaborn",
    "pandas",
    # 추가적인 의존성이 있다면 여기에 추가하세요
]

for imp in hidden_imports:
    pyinstaller_args.append(f"--hidden-import={imp}")

# 9. PyInstaller 실행
print("PyInstaller 실행 중... 이 과정은 몇 분 정도 소요될 수 있습니다.")
PyInstaller.__main__.run(pyinstaller_args)

print("빌드 완료! 실행 파일은 'dist' 폴더에 있습니다.")