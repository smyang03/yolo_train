# src/core/yolo_trainer.py - YOLOv7 훈련 관리 핵심 모듈

import subprocess
import threading
import time
import json
import os
import sys
import signal
import traceback
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime
import yaml
import re

# 로그 파서 import
from core.log_parser import YOLOv7LogParser

class YOLOv7Trainer:
    """YOLOv7 훈련 프로세스 관리 클래스"""

    def __init__(self):
        self.setup_paths()
        self.reset_state()
        self.log_parser = YOLOv7LogParser()
        self.callbacks = {}
        
    def setup_paths(self):
        """경로 설정 - 같은 레벨에 있는 YOLOv7 찾기"""
        self.app_dir = Path(__file__).parent.parent.parent  # yolov7_gui_standalone/
        self.project_workspace = self.app_dir.parent       # workspace/

        # YOLOv7 원본 경로 (같은 레벨) - 동적 경로 탐색
        # 우선순위: 1) 같은 부모 디렉토리, 2) 환경변수, 3) 현재 디렉토리
        yolo_candidates = [
            self.project_workspace / "yolov7",  # workspace/yolov7/
            self.app_dir.parent / "yolov7",      # 같은 레벨
            Path.cwd() / "yolov7",               # 현재 디렉토리
            Path.cwd().parent / "yolov7",        # 상위 디렉토리
        ]

        # 환경 변수 체크
        if os.environ.get('YOLOV7_PATH'):
            yolo_candidates.insert(0, Path(os.environ['YOLOV7_PATH']))

        self.yolo_original_dir = None
        for candidate in yolo_candidates:
            if candidate.exists() and (candidate / "train.py").exists():
                self.yolo_original_dir = candidate
                break

        # 찾지 못한 경우 기본값 설정
        if self.yolo_original_dir is None:
            self.yolo_original_dir = self.project_workspace / "yolov7"

        self.train_script = self.yolo_original_dir / "train.py"
        
        # GUI 프로젝트 경로들
        self.embedded_dir = self.app_dir / "yolov7_embedded"
        self.output_dir = self.app_dir / "outputs"
        self.temp_dir = self.app_dir / "temp"
        
        # 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # 경로 검증
        self.validate_paths()
        
    def validate_paths(self):
        """YOLOv7 경로 검증"""
        if not self.yolo_original_dir.exists():
            raise FileNotFoundError(
                f"YOLOv7 레포지토리를 찾을 수 없습니다: {self.yolo_original_dir}\n"
                f"workspace/ 폴더에 yolov7/ 가 있는지 확인하세요."
            )
        
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"train.py 파일을 찾을 수 없습니다: {self.train_script}"
            )
        
        print(f"✅ YOLOv7 경로 확인: {self.yolo_original_dir}")
        
    def reset_state(self):
        """훈련 상태 초기화"""
        self.process = None
        self.is_training = False
        self.is_paused = False
        self.current_metrics = {}
        self.training_config = {}
        self.start_time = None
        self.log_queue = Queue(maxsize=1000)  # 메모리 누수 방지: 최대 1000개 로그
        self.monitor_thread = None
        self.log_file_path = None
        self._stop_event = threading.Event()  # 스레드 안전 종료용
        
    def register_callback(self, event, callback):
        """이벤트 콜백 등록
        
        Args:
            event (str): 'metrics_update', 'training_complete', 'error' 등
            callback (function): 콜백 함수
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def trigger_callback(self, event, data=None):
        """콜백 실행"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"콜백 실행 오류 ({event}): {e}")
    
    def build_command(self, config):
        """YOLOv7 훈련 명령어 구성 - 하이퍼파라미터는 YAML 파일로만 처리"""
        python_exe = sys.executable

        # 🔥 workers=0 방지 (persistent_workers 오류 해결)
        workers = config.get("workers", 8)
        if workers == 0:
            workers = 1
            print("⚠️ workers=0은 YOLOv7에서 오류를 일으킵니다. 자동으로 1로 조정합니다.")

        # 기본 명령어 (하이퍼파라미터 값 제외)
        cmd = [
            python_exe,
            str(self.train_script),
            "--data", str(config["dataset_path"]),
            "--cfg", str(config["model_config"]),
            "--epochs", str(config["epochs"]),
            "--batch-size", str(config["batch_size"]),
            "--img-size", str(config["image_size"]),
            "--device", config["device"],
            "--project", str(self.output_dir),
            "--name", config["experiment_name"],
            "--workers", str(workers)
        ]
        
        # 가중치 파일 (선택사항)
        if config.get("weights_path"):
            cmd.extend(["--weights", str(config["weights_path"])])
        
        # 🔥 하이퍼파라미터 파일 처리
        hyp_file = config.get("hyperparams_file")
        if hyp_file:
            # 사용자가 지정한 하이퍼파라미터 파일 사용
            cmd.extend(["--hyp", str(hyp_file)])
            print(f"📄 사용자 지정 하이퍼파라미터 파일: {hyp_file}")
        else:
            # 하이퍼파라미터 파일이 없으면 YOLOv7 기본값 사용 (--hyp 옵션 생략)
            print("📄 YOLOv7 기본 하이퍼파라미터 사용")
        
        # 훈련 옵션들 (하이퍼파라미터와 별개)
        if config.get("cache_images"):
            cmd.append("--cache-images")
        if config.get("image_weights"):
            cmd.append("--image-weights")
        if config.get("multi_scale"):
            cmd.append("--multi-scale")
        if config.get("single_cls"):
            cmd.append("--single-cls")
        if config.get("adam"):
            cmd.append("--adam")
        if config.get("sync_bn"):
            cmd.append("--sync-bn")
        if config.get("rect"):
            cmd.append("--rect")

        # 🔥 메모리 최적화 옵션 (CUDA OOM 해결)
        # YOLOv7은 자체적으로 AMP를 지원하지 않지만, 수동으로 추가 가능
        # 일부 YOLOv7 버전은 내장 AMP 지원
        if config.get("mixed_precision", False):
            # YOLOv7의 일부 fork는 --amp 플래그를 지원
            # 지원하지 않으면 무시됨 (에러 없음)
            try:
                cmd.append("--amp")
                print("🔥 Mixed Precision (AMP) 활성화 - 메모리 50% 절약!")
            except:
                pass
        
        # 추가 훈련 옵션들
        if config.get("notest", False):
            cmd.append("--notest")
        if config.get("evolve", False):
            cmd.append("--evolve")
        if config.get("resume"):
            cmd.extend(["--resume", str(config["resume"])])
        
        return cmd
    
    def start_training(self, config):
        """훈련 시작"""
        if self.is_training:
            raise RuntimeError("이미 훈련이 진행 중입니다.")

        # ✨ 중요: 이전 훈련의 stop 이벤트 초기화 (Stop 후 재시작 시 필수)
        self._stop_event.clear()

        # 🔥 메모리 최적화 환경변수 설정 (CUDA OOM 해결)
        if config.get("memory_optimize", False):
            # CUDA 메모리 fragmentation 방지
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            print("🔥 메모리 Fragmentation 방지 활성화!")
            print("   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")

        self.training_config = config.copy()
        self.start_time = time.time()

        # 명령어 구성
        cmd = self.build_command(config)

        # Python unbuffered output 모드 추가 (stdout 버퍼링 방지)
        if cmd[0] == 'python':
            cmd.insert(1, '-u')

        print("🚀 YOLOv7 훈련 시작...")
        print(f"명령어: {' '.join(cmd)}")

        try:
            # 디버그 모드 확인 (환경변수 또는 기본값)
            debug_mode = os.getenv('YOLO_DEBUG', 'False').lower() == 'true'

            # 프로세스 시작
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # stderr 분리하여 에러 메시지 캡처
                universal_newlines=True,
                cwd=self.yolo_original_dir,  # YOLOv7 디렉토리에서 실행
                bufsize=0,  # 0 = unbuffered (즉시 출력)
                creationflags=0 if debug_mode or os.name != 'nt' else subprocess.CREATE_NO_WINDOW
            )

            # ✨ 프로세스 시작 확인 (2초 대기 후 상태 체크)
            print("⏳ 프로세스 시작 확인 중...")
            time.sleep(2)

            return_code = self.process.poll()
            if return_code is not None:
                # 프로세스가 즉시 종료됨!
                stderr_output = self.process.stderr.read() if self.process.stderr else ""
                stdout_output = self.process.stdout.read() if self.process.stdout else ""

                error_msg = (
                    f"❌ 훈련 프로세스가 즉시 종료되었습니다.\n\n"
                    f"Return Code: {return_code}\n\n"
                    f"Stderr:\n{stderr_output}\n\n"
                    f"Stdout:\n{stdout_output}"
                )

                print(error_msg)
                self.trigger_callback('error', {'message': error_msg})
                self.is_training = False
                return

            print("✅ 프로세스가 정상적으로 시작되었습니다.")
            self.is_training = True

            # 로그 모니터링 스레드 시작
            self.monitor_thread = threading.Thread(target=self._monitor_training)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            self.trigger_callback('training_started', {'config': config})

        except Exception as e:
            error_msg = f"훈련 시작 실패: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.trigger_callback('error', {'message': error_msg})
            raise

    def get_available_hyperparams(self):
        """사용 가능한 하이퍼파라미터 파일 목록 반환"""
        hyp_files = []
        
        # YOLOv7 하이퍼파라미터 디렉토리 확인
        search_paths = [
            self.yolo_original_dir / "data",
            self.yolo_original_dir / "cfg",
            self.yolo_original_dir,
            Path("data"),
            Path("cfg")
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for hyp_file in search_path.glob("hyp*.yaml"):
                    hyp_files.append({
                        'name': hyp_file.name,
                        'path': str(hyp_file),
                        'description': self.get_hyp_description(hyp_file.name),
                        'relative_path': str(hyp_file.relative_to(self.yolo_original_dir)) if hyp_file.is_relative_to(self.yolo_original_dir) else str(hyp_file)
                    })
        
        return hyp_files
    
    def create_custom_hyperparams_file(self, learning_rate=0.01, momentum=0.937, weight_decay=0.0005, 
                                    warmup_epochs=3.0, experiment_name="custom"):
        """사용자 정의 하이퍼파라미터 파일 생성"""
        
        # YOLOv7 기본 하이퍼파라미터 템플릿
        hyperparams = {
            # Learning rate settings
            'lr0': learning_rate,  # initial learning rate
            'lrf': 0.1,   # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': momentum,  # SGD momentum/Adam beta1
            'weight_decay': weight_decay,  # optimizer weight decay 5e-4
            'warmup_epochs': warmup_epochs,  # warmup epochs (fractions ok)
            'warmup_momentum': 0.8,  # warmup initial momentum
            'warmup_bias_lr': 0.1,  # warmup initial bias lr
            
            # Loss settings
            'box': 0.05,  # box loss gain
            'cls': 0.3,   # cls loss gain
            'cls_pw': 1.0,  # cls BCELoss positive_weight
            'obj': 0.7,   # obj loss gain (scale with pixels)
            'obj_pw': 1.0,  # obj BCELoss positive_weight
            'iou_t': 0.20,  # IoU training threshold
            'anchor_t': 4.0,  # anchor-multiple threshold
            'anchors': 3,  # anchors per output layer (0 to ignore)
            'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
            
            # Data augmentation
            'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,   # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,   # image HSV-Value augmentation (fraction)
            'degrees': 0.0,  # image rotation (+/- deg)
            'translate': 0.1,  # image translation (+/- fraction)
            'scale': 0.9,   # image scale (+/- gain)
            'shear': 0.0,   # image shear (+/- deg)
            'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
            'flipud': 0.0,  # image flip up-down (probability)
            'fliplr': 0.5,  # image flip left-right (probability)
            'mosaic': 1.0,  # image mosaic (probability)
            'mixup': 0.1,   # image mixup (probability)
            'copy_paste': 0.1,  # segment copy-paste (probability)
            'paste_in': 0.1,  # segment copy-paste (probability)
        }
        
        # 파일 저장
        import yaml
        
        hyp_file = self.temp_dir / f"hyp_custom_{experiment_name}.yaml"
        
        with open(hyp_file, 'w') as f:
            # 주석과 함께 저장
            f.write(f"# Custom YOLOv7 Hyperparameters - {experiment_name}\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            yaml.dump(hyperparams, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 커스텀 하이퍼파라미터 파일 생성: {hyp_file}")
        return hyp_file
    
    def get_hyp_description(self, filename):
        """하이퍼파라미터 파일 설명 반환"""
        descriptions = {
            'hyp.scratch.p5.yaml': '🎯 Default P5 (Small/Medium models) - Recommended',
            'hyp.scratch.p6.yaml': '🔥 P6 Large models (1280px) - High accuracy',
            'hyp.finetune.yaml': '⚡ Fine-tuning - For pretrained models',
            'hyp.Objects365.yaml': '📦 Objects365 dataset optimized',
            'hyp.scratch.low.yaml': '💚 Low resource training',
            'hyp.scratch.med.yaml': '📊 Medium resource training',
            'hyp.scratch.high.yaml': '🚀 High resource training',
        }
        
        return descriptions.get(filename, '📝 Custom hyperparameters')
    def _monitor_training(self):
        """훈련 모니터링 (별도 스레드) - 안전성 강화"""
        stderr_thread = None
        try:
            # stderr 모니터링 스레드 시작 (별도)
            def monitor_stderr():
                while self.is_training and self.process:
                    try:
                        if self.process.stderr:
                            line = self.process.stderr.readline()
                            if line:
                                line = line.strip()
                                if line:
                                    print(f"[STDERR] {line}")
                                    self.trigger_callback('log_update', {'line': f"⚠️ {line}"})
                    except:
                        break

            stderr_thread = threading.Thread(target=monitor_stderr)
            stderr_thread.daemon = True
            stderr_thread.start()

            # stdout 모니터링
            while self.is_training and self.process and not self._stop_event.is_set():
                try:
                    # ✨ 먼저 프로세스 상태 확인
                    if self.process.poll() is not None:
                        print("프로세스가 종료되었습니다.")
                        break

                    # stdout에서 한 줄씩 읽기
                    line = self.process.stdout.readline()

                    # ✨ EOF이고 프로세스가 종료된 경우만 break
                    if not line:
                        if self.process.poll() is not None:
                            break
                        else:
                            # 프로세스는 살아있지만 출력이 없음 (대기)
                            time.sleep(0.1)
                            continue

                    line = line.strip()
                    if line:
                        # 로그 파싱
                        parse_result = self.log_parser.parse_line(line)
                        if parse_result:
                            result_type = parse_result.get('type')
                            result_data = parse_result.get('data', {})

                            if result_type == 'metrics':
                                # 전체 메트릭 업데이트
                                self.current_metrics.update(result_data)
                                self.trigger_callback('metrics_update', result_data)
                            elif result_type == 'epoch':
                                # Epoch 정보 업데이트
                                self.current_metrics.update(result_data)
                                self.trigger_callback('epoch_update', result_data)
                            elif result_type == 'progress':
                                # 진행률 업데이트
                                self.trigger_callback('progress_update', result_data)

                        # 로그 큐에 추가 (큐가 가득 차면 오래된 항목 제거)
                        try:
                            self.log_queue.put(line, block=False)
                        except:
                            # 큐가 가득 차면 하나 제거하고 추가
                            try:
                                self.log_queue.get_nowait()
                                self.log_queue.put(line, block=False)
                            except:
                                pass

                        self.trigger_callback('log_update', {'line': line})

                except Exception as e:
                    if self.is_training:  # 정상 종료가 아닌 경우만 오류 보고
                        print(f"모니터링 오류: {e}")
                        self.trigger_callback('error', {'message': f"모니터링 오류: {e}"})
                    break

            # 프로세스 종료 확인
            if self.process:
                return_code = self.process.poll()
                if return_code is not None:
                    self.is_training = False
                    if return_code == 0:
                        self.trigger_callback('training_complete', {'success': True})
                    else:
                        # stderr 내용 읽기
                        if self.process.stderr:
                            stderr_remaining = self.process.stderr.read()
                            if stderr_remaining:
                                print(f"[STDERR 최종]: {stderr_remaining}")

                        self.trigger_callback('training_complete', {
                            'success': False,
                            'return_code': return_code
                        })

        finally:
            # 스레드 종료 시 리소스 정리
            print("모니터링 스레드 종료")
    
    def pause_training(self):
        """훈련 일시정지"""
        if not self.is_training or not self.process:
            return False
        
        try:
            if os.name == 'nt':  # Windows
                self.process.send_signal(signal.CTRL_C_EVENT)
            else:  # Unix/Linux
                self.process.send_signal(signal.SIGTERM)
            
            self.is_paused = True
            self.trigger_callback('training_paused')
            return True
            
        except Exception as e:
            self.trigger_callback('error', {'message': f"일시정지 실패: {e}"})
            return False
    
    def stop_training(self):
        """훈련 정지 - 리소스 안전 정리"""
        if not self.process:
            return True

        try:
            self.is_training = False
            self.is_paused = False
            self._stop_event.set()  # 모니터링 스레드에 종료 신호

            # 프로세스 종료
            self.process.terminate()

            # 강제 종료 대기
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("⚠️ 프로세스 강제 종료")
                self.process.kill()
                self.process.wait()

            # stdout 명시적으로 닫기
            if self.process.stdout:
                self.process.stdout.close()

            # 모니터링 스레드 종료 대기
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                if self.monitor_thread.is_alive():
                    print("⚠️ 모니터링 스레드가 정상 종료되지 않음")

            self.process = None
            self.monitor_thread = None
            self.trigger_callback('training_stopped')

            return True

        except Exception as e:
            self.trigger_callback('error', {'message': f"정지 실패: {e}"})
            return False

    def cleanup(self):
        """리소스 정리 - 애플리케이션 종료 시 호출"""
        print("🧹 YOLOv7Trainer 리소스 정리 중...")

        # 훈련 중이면 중지
        if self.is_training:
            self.stop_training()

        # 큐 비우기
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except:
                break

        print("✅ YOLOv7Trainer 정리 완료")
    
    def get_training_status(self):
        """훈련 상태 반환"""
        if not self.process:
            return "stopped"
        
        if self.is_paused:
            return "paused"
        elif self.is_training:
            return "training"
        else:
            return "stopping"
    
    def get_current_metrics(self):
        """현재 메트릭 반환"""
        return self.current_metrics.copy()
    
    def get_log_lines(self, max_lines=100):
        """로그 라인들 반환"""
        lines = []
        try:
            while not self.log_queue.empty() and len(lines) < max_lines:
                lines.append(self.log_queue.get_nowait())
        except Empty:
            pass
        return lines


# LogParser 클래스는 core/log_parser.py의 YOLOv7LogParser로 대체됨


# src/core/config_manager.py - 설정 관리 모듈

import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.app_dir / "resources" / "configs"
        self.default_config_path = self.config_dir / "default.yaml"
        
        self.config = self.load_default_config()
    
    def load_default_config(self) -> Dict[str, Any]:
        """기본 설정 로드"""
        default_config = {
            'dataset': {
                'mode': 'single',
                'path': '',
                'image_size': 640,
                'train_split': 0.8
            },
            'training': {
                'epochs': 300,
                'batch_size': 16,
                'learning_rate': 0.01,
                'workers': 8,
                'device': '0'
            },
            'model': {
                'config': 'cfg/training/yolov7.yaml',
                'weights': ''
            },
            'options': {
                'cache_images': False,
                'multi_scale': False,
                'image_weights': False,
                'rect': False,
                'adam': False,
                'sync_bn': False,
                'single_cls': False
            },
            'output': {
                'project_name': 'runs/train',
                'experiment_name': 'exp',
                'save_checkpoints': True
            }
        }
        
        # 파일에서 설정 로드 시도
        if self.default_config_path.exists():
            try:
                with open(self.default_config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self.merge_configs(default_config, file_config)
            except Exception as e:
                print(f"설정 파일 로드 실패: {e}")
        
        return default_config
    
    def merge_configs(self, base: Dict, override: Dict):
        """설정 병합"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self.merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get_training_config(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """UI 설정을 YOLOv7 훈련 설정으로 변환"""
        
        # 경로 설정
        app_dir = Path(__file__).parent.parent.parent
        yolo_dir = app_dir.parent / "yolov7"
        
        config = {
            'dataset_path': ui_config.get('dataset_path', ''),
            'model_config': yolo_dir / ui_config.get('model_config', 'cfg/training/yolov7.yaml'),
            'weights_path': ui_config.get('weights_path', ''),
            'epochs': ui_config.get('epochs', 300),
            'batch_size': ui_config.get('batch_size', 16),
            'image_size': ui_config.get('image_size', 640),
            'device': ui_config.get('device', '0'),
            'workers': ui_config.get('workers', 8),
            'learning_rate': ui_config.get('learning_rate', 0.01),
            'experiment_name': ui_config.get('experiment_name', 'exp'),
            
            # 옵션들
            'cache_images': ui_config.get('cache_images', False),
            'multi_scale': ui_config.get('multi_scale', False),
            'image_weights': ui_config.get('image_weights', False),
            'rect': ui_config.get('rect', False),
            'adam': ui_config.get('adam', False),
            'sync_bn': ui_config.get('sync_bn', False),
        }
        
        return config
    
    def save_config(self, config: Dict[str, Any], filepath: Path = None):
        """설정 저장"""
        if filepath is None:
            filepath = self.default_config_path
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"설정 저장 실패: {e}")
            return False


# src/core/model_manager.py - 모델 관리 모듈

import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class ModelManager:
    """훈련된 모델 관리 클래스"""
    
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent.parent
        self.output_dir = self.app_dir / "outputs"
        self.models_dir = self.app_dir / "saved_models"
        
        self.models_dir.mkdir(exist_ok=True)
        
        self.saved_models = []
        self.load_saved_models()
    
    def load_saved_models(self):
        """저장된 모델 목록 로드"""
        self.saved_models = []
        
        # outputs 폴더에서 모델 파일 찾기
        if self.output_dir.exists():
            for exp_dir in self.output_dir.iterdir():
                if exp_dir.is_dir():
                    weights_dir = exp_dir / "weights"
                    if weights_dir.exists():
                        self._scan_weights_directory(weights_dir, exp_dir.name)
    
    def _scan_weights_directory(self, weights_dir: Path, experiment_name: str):
        """weights 디렉토리 스캔"""
        for weight_file in weights_dir.glob("*.pt"):
            try:
                # 파일 정보 추출
                stat = weight_file.stat()
                
                model_info = {
                    'filepath': weight_file,
                    'filename': weight_file.name,
                    'experiment': experiment_name,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'created_time': datetime.fromtimestamp(stat.st_mtime),
                    'type': self._determine_model_type(weight_file.name)
                }
                
                self.saved_models.append(model_info)
                
            except Exception as e:
                print(f"모델 파일 정보 읽기 실패: {weight_file} - {e}")
    
    def _determine_model_type(self, filename: str) -> str:
        """파일명으로 모델 타입 결정"""
        if 'best' in filename.lower():
            return 'best'
        elif 'last' in filename.lower():
            return 'last'
        elif filename.startswith('epoch'):
            return 'checkpoint'
        else:
            return 'unknown'
    
    def get_best_models(self) -> Dict[str, Any]:
        """최고 성능 모델들 반환"""
        best_models = {
            'best_overall': None,
            'latest_best': None,
            'smallest_best': None
        }
        
        best_files = [m for m in self.saved_models if m['type'] == 'best']
        
        if best_files:
            # 최신 best 모델
            best_models['latest_best'] = max(best_files, key=lambda x: x['created_time'])
            
            # 가장 작은 best 모델
            best_models['smallest_best'] = min(best_files, key=lambda x: x['size_mb'])
            
            # 전체적으로 가장 좋은 모델 (최신을 기준으로)
            best_models['best_overall'] = best_models['latest_best']
        
        return best_models
    
    def copy_model_to_saved(self, model_info: Dict[str, Any], new_name: str = None) -> bool:
        """모델을 saved_models 폴더로 복사"""
        try:
            src_path = model_info['filepath']
            
            if new_name:
                dst_name = f"{new_name}.pt"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst_name = f"{model_info['experiment']}_{timestamp}.pt"
            
            dst_path = self.models_dir / dst_name
            
            shutil.copy2(src_path, dst_path)
            
            print(f"모델 복사 완료: {dst_path}")
            return True
            
        except Exception as e:
            print(f"모델 복사 실패: {e}")
            return False
    
    def delete_model(self, model_info: Dict[str, Any]) -> bool:
        """모델 파일 삭제"""
        try:
            model_info['filepath'].unlink()
            self.saved_models.remove(model_info)
            print(f"모델 삭제 완료: {model_info['filename']}")
            return True
            
        except Exception as e:
            print(f"모델 삭제 실패: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """모델 요약 정보 반환"""
        total_models = len(self.saved_models)
        total_size_mb = sum(m['size_mb'] for m in self.saved_models)
        
        by_type = {}
        for model in self.saved_models:
            model_type = model['type']
            if model_type not in by_type:
                by_type[model_type] = {'count': 0, 'size_mb': 0}
            by_type[model_type]['count'] += 1
            by_type[model_type]['size_mb'] += model['size_mb']
        
        return {
            'total_models': total_models,
            'total_size_mb': round(total_size_mb, 2),
            'by_type': by_type,
            'latest_model': max(self.saved_models, key=lambda x: x['created_time']) if self.saved_models else None
        }


# 사용 예시 및 테스트 코드
if __name__ == "__main__":
    # 테스트 코드
    print("🧪 YOLOv7 연결 모듈 테스트...")
    
    try:
        # YOLOv7 트레이너 초기화
        trainer = YOLOv7Trainer()
        
        # 콜백 등록
        def on_metrics_update(metrics):
            print(f"📊 메트릭 업데이트: {metrics}")
        
        def on_log_update(data):
            print(f"📝 로그: {data['line']}")
        
        trainer.register_callback('metrics_update', on_metrics_update)
        trainer.register_callback('log_update', on_log_update)
        
        print("✅ YOLOv7 연결 모듈 초기화 성공!")
        print(f"   YOLOv7 경로: {trainer.yolo_original_dir}")
        print(f"   출력 경로: {trainer.output_dir}")
        
        # 설정 관리자 테스트
        config_manager = ConfigManager()
        print("✅ 설정 관리자 초기화 성공!")
        
        # 모델 관리자 테스트
        model_manager = ModelManager()
        summary = model_manager.get_model_summary()
        print(f"✅ 모델 관리자 초기화 성공! 저장된 모델: {summary['total_models']}개")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("YOLOv7 레포지토리 경로를 확인하세요.")