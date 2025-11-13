# YOLOv7 Training GUI

YOLOv7 객체 탐지 모델 훈련을 위한 GUI 애플리케이션

## 설치 방법

1. 가상환경 생성
```bash
python -m venv venv
venv\Scripts\activate  # Windows
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
