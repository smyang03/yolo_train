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

## 📦 빠른 시작

```bash
# 1. Conda 환경 생성
conda env create -f environment.yaml
conda activate yolov7

# 2. GUI 실행
cd yolov7_gui_standalone
python main.py

# 3. EXE 빌드 (배포용)
python build_exe.py onedir
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
