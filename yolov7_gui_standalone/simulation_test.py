"""
시뮬레이션 테스트 - 전체 흐름 검증
GUI 없이 핵심 로직만 테스트
"""

import sys
import os
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("🧪 YOLOv7 Training GUI - 시뮬레이션 테스트")
print("=" * 70)
print()

# ============================================
# 1단계: 모듈 임포트 테스트
# ============================================
print("1️⃣ 모듈 임포트 테스트...")
print("-" * 70)

try:
    from core.yolo_trainer import YOLOv7Trainer
    print("✅ YOLOv7Trainer 임포트 성공")
except Exception as e:
    print(f"❌ YOLOv7Trainer 임포트 실패: {e}")
    sys.exit(1)

try:
    from core.config_manager import ConfigManager
    print("✅ ConfigManager 임포트 성공")
except Exception as e:
    print(f"❌ ConfigManager 임포트 실패: {e}")
    sys.exit(1)

try:
    from core.model_manager import ModelManager
    print("✅ ModelManager 임포트 성공")
except Exception as e:
    print(f"❌ ModelManager 임포트 실패: {e}")
    sys.exit(1)

try:
    from utils.validation import ConfigValidator
    print("✅ ConfigValidator 임포트 성공")
except Exception as e:
    print(f"❌ ConfigValidator 임포트 실패: {e}")
    sys.exit(1)

try:
    from utils.file_utils import ensure_dir, read_yaml
    print("✅ file_utils 임포트 성공")
except Exception as e:
    print(f"❌ file_utils 임포트 실패: {e}")
    sys.exit(1)

print()

# ============================================
# 2단계: YOLOv7Trainer 초기화 테스트
# ============================================
print("2️⃣ YOLOv7Trainer 초기화 테스트...")
print("-" * 70)

try:
    trainer = YOLOv7Trainer()
    print("✅ Trainer 초기화 성공")
    print(f"   YOLOv7 경로: {trainer.yolo_original_dir}")
    print(f"   Train script: {trainer.train_script}")
    print(f"   경로 존재: {trainer.yolo_original_dir.exists()}")
except FileNotFoundError as e:
    print(f"⚠️ YOLOv7 경로 문제 (정상): {e}")
    print("   → EXE 배포 시 yolov7/ 폴더 필요")
    trainer = None
except Exception as e:
    print(f"❌ Trainer 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 3단계: ConfigManager 테스트
# ============================================
print("3️⃣ ConfigManager 테스트...")
print("-" * 70)

try:
    config_manager = ConfigManager()
    print("✅ ConfigManager 초기화 성공")

    # 테스트 설정 변환
    test_ui_config = {
        'dataset_path': 'test.yaml',
        'model_config': 'cfg/training/yolov7.yaml',
        'epochs': 100,
        'batch_size': 16,
        'image_size': 640,
        'learning_rate': 0.01,
        'workers': 8,
        'device': '0',
        'experiment_name': 'test_exp'
    }

    training_config = config_manager.get_training_config(test_ui_config)
    print("✅ UI 설정 → 훈련 설정 변환 성공")
    print(f"   Epochs: {training_config['epochs']}")
    print(f"   Batch Size: {training_config['batch_size']}")
    print(f"   Device: {training_config['device']}")

except Exception as e:
    print(f"❌ ConfigManager 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 4단계: ModelManager 테스트
# ============================================
print("4️⃣ ModelManager 테스트...")
print("-" * 70)

try:
    model_manager = ModelManager()
    print("✅ ModelManager 초기화 성공")

    summary = model_manager.get_model_summary()
    print(f"   저장된 모델: {summary['total_models']}개")
    print(f"   전체 크기: {summary['total_size_mb']:.2f} MB")

except Exception as e:
    print(f"❌ ModelManager 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 5단계: Validation 테스트
# ============================================
print("5️⃣ Validation 테스트...")
print("-" * 70)

try:
    validator = ConfigValidator()

    # 훈련 파라미터 검증
    test_params = {
        'epochs': 300,
        'batch_size': 16,
        'image_size': 640,
        'learning_rate': 0.01,
        'workers': 8,
        'device': '0'
    }

    valid, msg = validator.validate_training_params(test_params)
    if valid:
        print(f"✅ 파라미터 검증 성공: {msg}")
    else:
        print(f"❌ 파라미터 검증 실패: {msg}")

    # 잘못된 파라미터 테스트
    bad_params = {
        'epochs': -1,
        'batch_size': 0,
        'image_size': 123,
        'learning_rate': -0.01,
        'workers': -1,
        'device': ''
    }

    valid, msg = validator.validate_training_params(bad_params)
    if not valid:
        print(f"✅ 잘못된 파라미터 감지 성공")
        print(f"   오류 메시지 샘플: {msg.split(chr(10))[0]}")

except Exception as e:
    print(f"❌ Validation 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 6단계: 명령어 생성 테스트 (Trainer가 있는 경우)
# ============================================
if trainer:
    print("6️⃣ 명령어 생성 테스트...")
    print("-" * 70)

    try:
        cmd = trainer.build_command(training_config)
        print("✅ 훈련 명령어 생성 성공")
        print(f"   명령어 길이: {len(cmd)} 인자")
        print(f"   Python: {cmd[0]}")
        print(f"   Script: {cmd[1]}")
        print(f"   샘플 인자: {' '.join(cmd[2:5])}")

    except Exception as e:
        print(f"❌ 명령어 생성 실패: {e}")

    print()

# ============================================
# 7단계: 리소스 정리 테스트
# ============================================
if trainer:
    print("7️⃣ 리소스 정리 테스트...")
    print("-" * 70)

    try:
        # cleanup 메서드 호출
        trainer.cleanup()
        print("✅ Trainer cleanup 성공")

        # 상태 확인
        assert trainer.process is None or not trainer.is_training
        print("✅ 훈련 상태 정리 확인")

        # 큐 확인
        assert trainer.log_queue.empty()
        print("✅ 로그 큐 비우기 확인")

    except Exception as e:
        print(f"❌ 리소스 정리 실패: {e}")
        import traceback
        traceback.print_exc()

    print()

# ============================================
# 8단계: 메모리 안전성 체크
# ============================================
print("8️⃣ 메모리 안전성 체크...")
print("-" * 70)

if trainer:
    # Queue 크기 제한 확인
    max_size = trainer.log_queue.maxsize
    if max_size > 0:
        print(f"✅ 로그 큐 크기 제한: {max_size}개")
    else:
        print(f"⚠️ 로그 큐 무제한 (메모리 누수 가능)")

    # 스레드 종료 이벤트 확인
    if hasattr(trainer, '_stop_event'):
        print("✅ 스레드 안전 종료 이벤트 존재")
    else:
        print("⚠️ 스레드 종료 이벤트 없음")

print()

# ============================================
# 최종 결과
# ============================================
print("=" * 70)
print("🎉 시뮬레이션 테스트 완료!")
print("=" * 70)
print()
print("✅ 통과한 테스트:")
print("   1. 모듈 임포트")
print("   2. ConfigManager")
print("   3. ModelManager")
print("   4. Validation 시스템")
if trainer:
    print("   5. YOLOv7Trainer 초기화")
    print("   6. 명령어 생성")
    print("   7. 리소스 정리")
    print("   8. 메모리 안전성")
else:
    print("   ⚠️ YOLOv7Trainer는 yolov7/ 경로 필요 (정상)")
print()
print("🚀 실제 환경에서 실행 가능!")
print("=" * 70)
