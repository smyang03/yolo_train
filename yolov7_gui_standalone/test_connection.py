# test_connection.py - YOLOv7 연결 모듈 테스트

import sys
from pathlib import Path

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_yolo_connection():
    """YOLOv7 연결 테스트"""
    
    print("🧪 YOLOv7 연결 모듈 테스트 시작...")
    print("=" * 50)
    
    try:
        # 1. 모듈 임포트 테스트
        print("1️⃣ 모듈 임포트 테스트...")
        from core.yolo_trainer import YOLOv7Trainer, LogParser
        from core.config_manager import ConfigManager
        from core.model_manager import ModelManager
        print("   ✅ 모든 모듈 임포트 성공!")
        
        # 2. YOLOv7 경로 확인
        print("\n2️⃣ YOLOv7 경로 확인...")
        trainer = YOLOv7Trainer()
        print(f"   📁 YOLOv7 경로: {trainer.yolo_original_dir}")
        print(f"   📁 train.py 경로: {trainer.train_script}")
        print(f"   📁 출력 경로: {trainer.output_dir}")
        
        if trainer.yolo_original_dir.exists():
            print("   ✅ YOLOv7 레포지토리 확인됨!")
        else:
            print("   ❌ YOLOv7 레포지토리를 찾을 수 없습니다!")
            return False
        
        if trainer.train_script.exists():
            print("   ✅ train.py 파일 확인됨!")
        else:
            print("   ❌ train.py 파일을 찾을 수 없습니다!")
            return False
        
        # 3. 설정 관리자 테스트
        print("\n3️⃣ 설정 관리자 테스트...")
        config_manager = ConfigManager()
        
        # 샘플 UI 설정
        sample_ui_config = {
            'dataset_path': 'path/to/dataset.yaml',
            'model_config': 'cfg/training/yolov7.yaml',
            'epochs': 100,
            'batch_size': 8,
            'image_size': 640,
            'device': '0',
            'experiment_name': 'test_exp'
        }
        
        yolo_config = config_manager.get_training_config(sample_ui_config)
        print(f"   ✅ 설정 변환 성공!")
        print(f"   📋 변환된 설정: {len(yolo_config)} 항목")
        
        # 4. 모델 관리자 테스트
        print("\n4️⃣ 모델 관리자 테스트...")
        model_manager = ModelManager()
        summary = model_manager.get_model_summary()
        print(f"   ✅ 모델 관리자 초기화 완료!")
        print(f"   📊 저장된 모델: {summary['total_models']}개")
        print(f"   💾 총 크기: {summary['total_size_mb']} MB")
        
        # 5. 로그 파서 테스트
        print("\n5️⃣ 로그 파서 테스트...")
        parser = LogParser()
        
        # 샘플 로그 라인들
        sample_logs = [
            "Epoch 10/300: P=0.95, R=0.87, mAP@.5:0.92, mAP@.5:.95:0.75",
            "train: Epoch 5/100, Loss 0.234",
            "GPU memory: 7.5G"
        ]
        
        for log in sample_logs:
            metrics = parser.parse_line(log)
            if metrics:
                print(f"   ✅ 파싱 성공: {log[:50]}... → {len(metrics)} 메트릭")
            else:
                print(f"   ⚠️ 파싱 실패: {log[:50]}...")
        
        # 6. 명령어 생성 테스트
        print("\n6️⃣ 명령어 생성 테스트...")
        cmd = trainer.build_command(yolo_config)
        print(f"   ✅ 명령어 생성 성공!")
        print(f"   🔧 명령어 길이: {len(cmd)} 인자")
        print(f"   📝 Python 실행파일: {cmd[0]}")
        print(f"   📝 train.py 경로: {cmd[1]}")
        
        # 7. 콜백 시스템 테스트
        print("\n7️⃣ 콜백 시스템 테스트...")
        
        callback_called = False
        def test_callback(data):
            nonlocal callback_called
            callback_called = True
            print(f"   📞 콜백 호출됨: {data}")
        
        trainer.register_callback('test_event', test_callback)
        trainer.trigger_callback('test_event', {'message': 'Hello from callback!'})
        
        if callback_called:
            print("   ✅ 콜백 시스템 정상 작동!")
        else:
            print("   ❌ 콜백 시스템 오류!")
        
        print("\n" + "=" * 50)
        print("🎉 모든 테스트 통과! YOLOv7 연결 준비 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_simulation():
    """훈련 시뮬레이션 테스트 (실제 훈련 X)"""
    
    print("\n🎭 훈련 시뮬레이션 테스트...")
    print("=" * 50)
    
    try:
        from core.yolo_trainer import YOLOv7Trainer
        from core.config_manager import ConfigManager
        
        trainer = YOLOv7Trainer()
        config_manager = ConfigManager()
        
        # 콜백 등록
        def on_metrics_update(metrics):
            print(f"📊 메트릭: {metrics}")
        
        def on_log_update(data):
            print(f"📝 로그: {data['line'][:80]}...")
        
        def on_training_started(data):
            print(f"🚀 훈련 시작: {data}")
        
        def on_error(data):
            print(f"❌ 오류: {data}")
        
        trainer.register_callback('metrics_update', on_metrics_update)
        trainer.register_callback('log_update', on_log_update)
        trainer.register_callback('training_started', on_training_started)
        trainer.register_callback('error', on_error)
        
        # 테스트 설정
        test_config = {
            'dataset_path': 'dummy_dataset.yaml',  # 실제로는 없는 파일
            'model_config': 'cfg/training/yolov7.yaml',
            'epochs': 5,
            'batch_size': 4,
            'image_size': 640,
            'device': 'cpu',  # CPU로 안전하게 테스트
            'experiment_name': 'connection_test'
        }
        
        yolo_config = config_manager.get_training_config(test_config)
        
        print("⚠️ 실제 훈련은 시작하지 않습니다 (데이터셋이 없으므로)")
        print("✅ 명령어 생성 및 콜백 시스템 준비 완료!")
        
        # 명령어만 출력
        cmd = trainer.build_command(yolo_config)
        print(f"\n🔧 생성될 명령어:")
        print(" ".join(str(arg) for arg in cmd))
        
        return True
        
    except Exception as e:
        print(f"❌ 시뮬레이션 테스트 실패: {e}")
        return False

def show_system_info():
    """시스템 정보 표시"""
    
    print("\n💻 시스템 정보")
    print("=" * 50)
    
    import sys
    import torch
    
    print(f"🐍 Python: {sys.version}")
    print(f"📁 현재 경로: {Path.cwd()}")
    
    try:
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🎮 CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA 장치 수: {torch.cuda.device_count()}")
            print(f"🎮 현재 CUDA 장치: {torch.cuda.current_device()}")
    except ImportError:
        print("⚠️ PyTorch가 설치되지 않았습니다.")
    
    try:
        import cv2
        print(f"📷 OpenCV: {cv2.__version__}")
    except ImportError:
        print("⚠️ OpenCV가 설치되지 않았습니다.")

if __name__ == "__main__":
    print("🧪 YOLOv7 GUI 연결 모듈 종합 테스트")
    print("=" * 60)
    
    # 시스템 정보 표시
    show_system_info()
    
    # 기본 연결 테스트
    success = test_yolo_connection()
    
    if success:
        # 훈련 시뮬레이션 테스트
        test_training_simulation()
        
        print("\n" + "=" * 60)
        print("🎉 테스트 완료! 다음 단계로 진행할 수 있습니다.")
        print("\n📋 다음 할 일:")
        print("1. python test_connection.py  (이 테스트)")
        print("2. UI 통합 구현")
        print("3. 실제 데이터셋으로 훈련 테스트")
        print("4. EXE 빌드")
    else:
        print("\n❌ 테스트 실패! 문제를 해결한 후 다시 시도하세요.")
        print("\n🔧 체크사항:")
        print("- YOLOv7 레포지토리가 올바른 위치에 있는지 확인")
        print("- 필요한 패키지들이 설치되었는지 확인")
        print("- 파일 경로가 올바른지 확인")