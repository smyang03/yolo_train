
import subprocess
import threading
import time
import json
import os
import sys
import signal
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime
import re

class YOLOv7Trainer:
    """YOLOv7 훈련 프로세스 관리 클래스"""
    
    def __init__(self):
        self.setup_paths()
        self.reset_state()
        self.log_parser = LogParser()
        self.callbacks = {}
        
    def setup_paths(self):
        """경로 설정 - 같은 레벨에 있는 YOLOv7 찾기"""
        self.app_dir = Path(__file__).parent.parent.parent  # yolov7_gui_standalone/
        self.project_workspace = self.app_dir.parent       # workspace/
        
        # YOLOv7 원본 경로 (같은 레벨)
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
                f"YOLOv7 레포지토리를 찾을 수 없습니다: {self.yolo_original_dir}\\n"
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
        self.log_queue = Queue()
        self.monitor_thread = None
        self.log_file_path = None
        
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
        """YOLOv7 훈련 명령어 구성"""
        python_exe = sys.executable
        
        cmd = [
            python_exe,
            str(self.train_script),
            "--data", str(config["dataset_path"]),
            "--cfg", str(config["model_config"]),
            "--epochs", str(config["epochs"]),
            "--batch-size", str(config["batch_size"]),
            "--img", str(config["image_size"]),
            "--device", config["device"],
            "--project", str(self.output_dir),
            "--name", config["experiment_name"],
            "--workers", str(config.get("workers", 8))
        ]
        
        # 가중치 파일 (선택사항)
        if config.get("weights_path"):
            cmd.extend(["--weights", str(config["weights_path"])])
        
        # 추가 옵션들
        if config.get("cache_images"):
            cmd.append("--cache")
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
        
        # 학습률
        if config.get("learning_rate"):
            cmd.extend(["--lr0", str(config["learning_rate"])])
        
        return cmd
    
    def start_training(self, config):
        """훈련 시작"""
        if self.is_training:
            raise RuntimeError("이미 훈련이 진행 중입니다.")
        
        self.training_config = config.copy()
        self.start_time = time.time()
        
        # 명령어 구성
        cmd = self.build_command(config)
        
        print("🚀 YOLOv7 훈련 시작...")
        print(f"명령어: {' '.join(cmd)}")
        
        try:
            # 프로세스 시작
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=self.yolo_original_dir,  # YOLOv7 디렉토리에서 실행
                bufsize=1,  # 라인 버퍼링
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            self.is_training = True
            
            # 로그 모니터링 스레드 시작
            self.monitor_thread = threading.Thread(target=self._monitor_training)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.trigger_callback('training_started', {'config': config})
            
        except Exception as e:
            self.trigger_callback('error', {'message': f"훈련 시작 실패: {e}"})
            raise
    
    def _monitor_training(self):
        """훈련 모니터링 (별도 스레드)"""
        while self.is_training and self.process:
            try:
                # stdout에서 한 줄씩 읽기
                line = self.process.stdout.readline()
                
                if not line:
                    break
                
                line = line.strip()
                if line:
                    # 로그 파싱
                    metrics = self.log_parser.parse_line(line)
                    if metrics:
                        self.current_metrics.update(metrics)
                        self.trigger_callback('metrics_update', self.current_metrics)
                    
                    # 로그 큐에 추가
                    self.log_queue.put(line)
                    self.trigger_callback('log_update', {'line': line})
                
            except Exception as e:
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
                    self.trigger_callback('training_complete', {'success': False, 'return_code': return_code})
    
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
        """훈련 정지"""
        if not self.process:
            return True
        
        try:
            self.is_training = False
            self.is_paused = False
            
            # 프로세스 종료
            self.process.terminate()
            
            # 강제 종료 대기
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            
            self.process = None
            self.trigger_callback('training_stopped')
            
            return True
            
        except Exception as e:
            self.trigger_callback('error', {'message': f"정지 실패: {e}"})
            return False
    
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


class LogParser:
    """YOLOv7 로그 파싱 클래스"""
    
    def __init__(self):
        self.patterns = {
            'epoch': re.compile(r'Epoch\\s+(\\d+)/(\\d+)'),
            'metrics': re.compile(r'P:\\s*([\\d.]+)\\s+R:\\s*([\\d.]+)\\s+mAP@\\.5:\\s*([\\d.]+)\\s+mAP@\\.5:.95:\\s*([\\d.]+)'),
            'loss': re.compile(r'train.*?(\\d+\\.\\d+)'),
            'lr': re.compile(r'lr:\\s*([\\d.e-]+)'),
            'gpu_memory': re.compile(r'(\\d+\\.?\\d*)G'),
            'time': re.compile(r'(\\d+:\\d+:\\d+)'),
        }
    
    def parse_line(self, line):
        """로그 라인 파싱"""
        metrics = {}
        
        # Epoch 정보
        epoch_match = self.patterns['epoch'].search(line)
        if epoch_match:
            metrics['current_epoch'] = int(epoch_match.group(1))
            metrics['total_epochs'] = int(epoch_match.group(2))
        
        # 성능 메트릭
        metrics_match = self.patterns['metrics'].search(line)
        if metrics_match:
            metrics.update({
                'precision': float(metrics_match.group(1)),
                'recall': float(metrics_match.group(2)),
                'map50': float(metrics_match.group(3)),
                'map95': float(metrics_match.group(4))
            })
        
        # Loss
        loss_match = self.patterns['loss'].search(line)
        if loss_match:
            metrics['loss'] = float(loss_match.group(1))
        
        # Learning Rate
        lr_match = self.patterns['lr'].search(line)
        if lr_match:
            metrics['learning_rate'] = float(lr_match.group(1))
        
        # GPU 메모리
        gpu_match = self.patterns['gpu_memory'].search(line)
        if gpu_match:
            metrics['gpu_memory'] = f"{gpu_match.group(1)}G"
        
        return metrics if metrics else None


# 테스트 함수
if __name__ == "__main__":
    print("🧪 YOLOv7Trainer 테스트...")
    
    try:
        trainer = YOLOv7Trainer()
        
        # 콜백 테스트
        def test_callback(data):
            print(f"콜백 테스트: {data}")
        
        trainer.register_callback('test', test_callback)
        trainer.trigger_callback('test', "Hello!")
        
        print("✅ YOLOv7Trainer 초기화 성공!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")