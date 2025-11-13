# YOLOv7 Training GUI - EXE 사용 가이드

## 📦 배포 패키지 구성

배포 시 다음과 같은 구조로 제공하세요:

```
YOLOv7_Training_Package/
├── YOLOv7_Training_GUI/     # EXE 및 DLL들
│   ├── YOLOv7_Training_GUI.exe
│   ├── python310.dll
│   ├── _internal/             # 의존성 파일들
│   └── ...
│
├── yolov7/                    # YOLOv7 원본 레포지토리 (필수!)
│   ├── train.py
│   ├── cfg/
│   ├── data/
│   ├── models/
│   └── utils/
│
└── README.txt                 # 이 가이드
```

---

## ⚙️ 실행 전 준비사항

### 1. YOLOv7 레포지토리 배치

**방법 A: 같은 폴더에 배치 (권장)**
```
YOLOv7_Training_GUI/
yolov7/  ← 이 폴더가 필요합니다!
```

**방법 B: 환경 변수 설정**
```cmd
set YOLOV7_PATH=C:\path\to\yolov7
```

### 2. 시스템 요구사항

**필수**:
- Windows 10 64-bit 이상
- 8GB RAM 이상
- 10GB 여유 공간

**권장** (GPU 사용 시):
- NVIDIA GPU (8GB VRAM 이상)
- CUDA 11.6
- cuDNN 8.x

### 3. GPU 사용 확인

GPU가 없거나 CUDA가 설치되지 않은 경우:
- Device 설정에서 `cpu` 선택
- 훈련 속도가 매우 느려집니다

---

## 🚀 실행 방법

### 첫 실행

1. `YOLOv7_Training_GUI.exe` 더블클릭
2. 초기 로딩 시간: 30초 ~ 2분 (PyTorch 초기화)
3. GUI 창이 열리면 준비 완료

### 데이터셋 준비

1. 데이터셋을 YOLO 형식으로 준비
2. `data.yaml` 파일 작성:

```yaml
train: path/to/train/images
val: path/to/valid/images
nc: 3  # 클래스 수
names: ['person', 'car', 'dog']
```

### 훈련 시작

1. **Dataset 섹션**
   - Browse 버튼으로 data.yaml 선택

2. **Model 섹션**
   - 모델 선택 (YOLOv7 변형)
   - 가중치 옵션 선택

3. **Training Parameters**
   - Epochs, Batch Size 등 설정
   - GPU 사용 시: Device = `0`
   - CPU 사용 시: Device = `cpu`

4. **🚀 Start Enhanced Training** 클릭

5. **진행사항 탭**에서 실시간 모니터링

---

## ⚠️ 문제 해결

### EXE가 시작되지 않음

**증상**: 더블클릭 후 아무 반응 없음

**해결**:
1. 명령 프롬프트에서 실행:
   ```cmd
   cd path\to\YOLOv7_Training_GUI
   YOLOv7_Training_GUI.exe
   ```
2. 오류 메시지 확인

### "YOLOv7 레포지토리를 찾을 수 없습니다" 오류

**원인**: yolov7 폴더를 찾을 수 없음

**해결**:
1. yolov7 폴더가 EXE와 같은 위치에 있는지 확인
2. 또는 환경 변수 설정:
   ```cmd
   set YOLOV7_PATH=C:\full\path\to\yolov7
   ```

### "CUDA out of memory" 오류

**원인**: GPU 메모리 부족

**해결**:
1. Batch Size 줄이기 (16 → 8 → 4)
2. Image Size 줄이기 (640 → 512)
3. Workers 수 줄이기
4. 또는 CPU 모드 사용

### GUI가 느리게 반응함

**원인**: 훈련 중 메모리 사용량 증가

**정상 동작**:
- 훈련 시작 후 시스템 리소스 사용 증가는 정상입니다
- 메트릭 차트 업데이트로 인한 일시적 지연 가능

---

## 💡 사용 팁

### 1. Best 모델 자동 저장

훈련 중 자동으로 최고 성능 모델이 추적됩니다:
- Best Precision
- Best Recall
- Best Balance (P+R)
- Best mAP

**모델 선택** 탭에서 확인 가능

### 2. 메모리 관리

장시간 훈련 시:
- 로그가 1000개로 제한됨 (메모리 누수 방지)
- 주기적으로 체크포인트 저장
- 모니터링 차트는 자동으로 관리됨

### 3. 안전한 종료

훈련 중 종료 시:
- "훈련 진행 중입니다. 정말로 종료하시겠습니까?" 확인 창
- Yes 선택 시 안전하게 프로세스 정리 후 종료
- 현재까지 훈련된 체크포인트는 보존됨

### 4. 멀티 GPU 사용

```
Device: 0,1      # GPU 0, 1 사용
Device: 0,1,2,3  # 4개 GPU 사용
```

---

## 📊 훈련 결과 확인

### 출력 파일 위치

```
yolov7_gui_standalone/outputs/
└── exp1/                    # 실험 이름
    ├── weights/
    │   ├── best.pt          # 최고 성능 모델
    │   ├── last.pt          # 마지막 에폭 모델
    │   └── epoch_*.pt       # 체크포인트들
    ├── results.png          # 훈련 그래프
    └── logs.txt             # 훈련 로그
```

### 모델 사용

1. **테스트**: GUI의 "🧪 Test Model" 버튼
2. **배포**: "🚀 Deploy Model" → Triton Server
3. **ONNX 변환**: "📦 Export ONNX" → 다른 프레임워크

---

## 🔧 고급 설정

### 하이퍼파라미터 커스터마이징

1. **Hyperparameters 섹션**에서 "Custom" 선택
2. YAML 파일 업로드 또는 GUI에서 직접 수정
3. Learning Rate, Momentum 등 조정

### 데이터셋 병합

1. **Dataset Mode**에서 "Multiple Datasets" 선택
2. 여러 데이터셋 YAML 추가
3. Merge 옵션 설정:
   - Shuffle: 데이터 섞기
   - Balance: 클래스 균형
   - Remove duplicates: 중복 제거

---

## 📞 지원

**문제 발생 시**:
1. `error.log` 파일 확인 (EXE와 같은 폴더)
2. GitHub Issues에 보고
3. 로그 파일 첨부

**시스템 정보 확인**:
- "🧪 Test Connection" 버튼으로 연결 테스트
- 콘솔 창에서 시스템 정보 확인

---

## ⚡ 성능 최적화

### GPU 사용 극대화

```
Batch Size: GPU 메모리에 맞게 최대화
Workers: CPU 코어 수 -2
Image Size: 640 (기본) 또는 512 (속도 우선)
```

### CPU 모드 최적화

```
Batch Size: 4-8
Workers: 2-4
Image Size: 416 (작게)
Epochs: 줄이기 (테스트용)
```

---

**버전**: 1.0
**최종 업데이트**: 2025

**Made with ❤️ for the Computer Vision Community**
