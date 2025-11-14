# YOLOv7 Training GUI - 완성된 배포 버전

> **전문적인 GUI 기반 YOLOv7 객체 탐지 모델 훈련 플랫폼**

YOLOv7 모델을 명령줄 없이 직관적인 그래픽 인터페이스로 훈련할 수 있는 독립 실행 가능한 데스크톱 애플리케이션입니다.

---

## ✨ 주요 기능

### 🎯 직관적인 GUI
- **완전한 시각화**: 모든 훈련 설정을 GUI에서 제어
- **실시간 모니터링**: Matplotlib 차트로 훈련 메트릭 실시간 표시
- **Best 모델 자동 추적**: Precision, Recall, Balance, mAP 기준

### 🤖 모델 관리
- **7가지 YOLOv7 변형**: Default, X, Tiny, W6, E6, D6, E6E
- **가중치 옵션**: 처음부터 / 공식 가중치 / 커스텀
- **클래스별 성능 분석**

### ⚙️ 하이퍼파라미터 설정
- YOLOv7 기본값 / 프리셋 / 커스텀 YAML
- 실시간 설정 검증

### 🚀 배포
- PyInstaller EXE 빌드
- Inno Setup 인스톨러
- ONNX 변환 지원

---

## 📦 실행 방법

### 방법 1: Python으로 직접 실행 (개발/테스트)

**환경 구성:**

```bash
# 1. YOLOv7 원본 레포지토리 클론 (필수!)
git clone https://github.com/WongKinYiu/yolov7.git

# 2. Conda 환경 생성
conda env create -f environment.yaml
conda activate yolov7

# 3. PyTorch 설치 (CUDA 11.6)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 4. 추가 의존성 설치
pip install matplotlib pyyaml opencv-python pillow tqdm
```

**실행:**

```bash
cd yolov7_gui_standalone
python main.py
```

**장점:**
- 즉시 코드 수정 및 테스트 가능
- 디버깅 용이
- 개발 환경에 적합

**단점:**
- Python 환경 필요
- 의존성 수동 관리 필요

---

### 방법 2: EXE 파일 빌드 (배포/사용자)

**EXE 빌드 과정:**

```bash
# 1. Python 환경이 준비된 상태에서
cd yolov7_gui_standalone

# 2. PyInstaller 설치
pip install pyinstaller

# 3. EXE 빌드 (단일 폴더 방식 - 권장)
python build_exe.py onedir

# 또는 단일 파일 방식 (느리지만 단일 EXE)
python build_exe.py onefile
```

**빌드 결과:**

```
dist/
└── YOLOv7_Training_GUI/     # 이 폴더를 배포
    ├── YOLOv7_Training_GUI.exe
    ├── _internal/           # 필요한 DLL, 라이브러리
    └── ...
```

**배포 패키지 구성:**

```
YOLOv7_Training_Package/
├── YOLOv7_Training_GUI/     # 빌드된 EXE 폴더
│   └── YOLOv7_Training_GUI.exe
└── yolov7/                   # YOLOv7 원본 (필수!)
    ├── train.py
    ├── test.py
    ├── models/
    ├── utils/
    └── cfg/
```

**실행:**
1. `YOLOv7_Training_Package` 폴더 전체를 사용자에게 전달
2. 사용자는 `YOLOv7_Training_GUI/YOLOv7_Training_GUI.exe` 더블 클릭
3. Python 설치 불필요!

**장점:**
- Python 설치 불필요
- 일반 사용자에게 편리
- 깔끔한 배포

**단점:**
- 빌드 시간 소요 (5-10분)
- 파일 크기 큼 (~500MB-1GB)
- 수정 시 재빌드 필요

---

## 📋 빠른 시작 요약

**개발자/테스트:**
```bash
conda activate yolov7
cd yolov7_gui_standalone
python main.py
```

**최종 사용자 배포:**
```bash
python build_exe.py onedir
# → dist/YOLOv7_Training_GUI/ 폴더 + yolov7/ 폴더 함께 배포
```

---

## 🗂️ 프로젝트 구조

```
yolo_train/
├── yolov7/                    # YOLOv7 원본
└── yolov7_gui_standalone/     # GUI 앱
    ├── main.py
    ├── build_exe.py           # EXE 빌드
    ├── src/
    │   ├── core/              # 훈련 로직
    │   ├── ui/                # GUI (2668줄)
    │   └── utils/             # 검증, 파일 유틸
    └── yolov7_embedded/
```

---

## 🛠️ 주요 개선 (v1.0)

✅ 동적 경로 탐색 (하드코딩 제거)
✅ 완전한 입력 검증 시스템
✅ 파일 유틸리티 모듈
✅ PyInstaller 빌드 시스템
✅ Best 모델 자동 추적
✅ 클래스별 성능 분석

---

## 📖 사용 가이드

### 데이터셋 준비

```yaml
# data.yaml
train: path/to/train/images
val: path/to/valid/images
nc: 3
names: ['person', 'car', 'dog']
```

### 훈련 시작

1. Dataset 섹션에서 YAML 선택
2. Model 선택 (YOLOv7 변형)
3. Hyperparameters 설정
4. "🚀 Start Training" 클릭
5. 실시간 모니터링 및 Best 모델 확인

---

## 💻 시스템 요구사항

**최소**: Windows 10, 8GB RAM, 10GB Storage
**권장**: Windows 11, 16GB RAM, NVIDIA GPU (8GB+ VRAM)

---

## 🐛 문제 해결

**YOLOv7 경로 오류**: `set YOLOV7_PATH=C:\path\to\yolov7`
**CUDA 메모리 부족**: Batch Size 줄이기

---

## 📚 문서

- [사용자 매뉴얼](yolov7_gui_standalone/docs/user_manual.md)
- [개발자 가이드](yolov7_gui_standalone/docs/developer_guide.md)

---

**Made with ❤️ for the Computer Vision Community**
