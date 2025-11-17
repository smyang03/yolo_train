"""
YOLOv7 로그 파서
훈련 로그에서 메트릭을 추출하는 모듈
"""

import sys
import io
import re
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == 'win32':
    try:
        if sys.version_info >= (3, 7):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


@dataclass
class TrainingMetrics:
    """훈련 메트릭 데이터 클래스"""
    epoch: int = 0
    total_epochs: int = 0
    gpu_mem: str = "0G"

    # Training losses
    box_loss: float = 0.0
    obj_loss: float = 0.0
    cls_loss: float = 0.0
    total_loss: float = 0.0

    # Training info
    instances: int = 0
    img_size: int = 640

    # Validation metrics
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0  # mAP@0.5
    map95: float = 0.0  # mAP@0.5:0.95

    # Validation losses
    val_box_loss: float = 0.0
    val_obj_loss: float = 0.0
    val_cls_loss: float = 0.0

    # Progress
    progress_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'gpu_mem': self.gpu_mem,
            'box_loss': self.box_loss,
            'obj_loss': self.obj_loss,
            'cls_loss': self.cls_loss,
            'total_loss': self.total_loss,
            'instances': self.instances,
            'img_size': self.img_size,
            'precision': self.precision,
            'recall': self.recall,
            'map50': self.map50,
            'map95': self.map95,
            'val_box_loss': self.val_box_loss,
            'val_obj_loss': self.val_obj_loss,
            'val_cls_loss': self.val_cls_loss,
            'progress_percent': self.progress_percent
        }


class YOLOv7LogParser:
    """YOLOv7 훈련 로그 파서"""

    def __init__(self):
        self.current_metrics = TrainingMetrics()
        self.last_epoch = 0

        # 정규식 패턴들
        # 예: "     0/299     2.59G   0.02872  0.007841  0.009021   0.04558         8       640     0.884    0.5205    0.6525     0.519   0.03446   0.02027   0.01341"
        self.metrics_pattern = re.compile(
            r'\s*(\d+)/(\d+)\s+' +  # epoch/total_epochs
            r'([\d.]+)G\s+' +  # GPU memory
            r'([\d.]+)\s+' +  # box_loss
            r'([\d.]+)\s+' +  # obj_loss
            r'([\d.]+)\s+' +  # cls_loss
            r'([\d.]+)\s+' +  # total_loss
            r'(\d+)\s+' +  # instances
            r'(\d+)\s+' +  # img_size
            r'([\d.]+)\s+' +  # precision
            r'([\d.]+)\s+' +  # recall
            r'([\d.]+)\s+' +  # mAP@0.5
            r'([\d.]+)\s+' +  # mAP@0.5:0.95
            r'([\d.]+)\s+' +  # val_box_loss
            r'([\d.]+)\s+' +  # val_obj_loss
            r'([\d.]+)'  # val_cls_loss
        )

        # 다른 형식의 epoch 표시: "Epoch 10/300"
        self.epoch_pattern = re.compile(r'Epoch\s+(\d+)/(\d+)', re.IGNORECASE)

        # 진행률 바: "100%|██████████| 15/15"
        self.progress_pattern = re.compile(r'(\d+)%\|')

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        로그 라인 파싱

        Args:
            line: 로그 라인

        Returns:
            파싱된 메트릭 딕셔너리 또는 None
        """
        line = line.strip()

        if not line:
            return None

        # 메트릭 라인 파싱 시도
        metrics_match = self.metrics_pattern.match(line)
        if metrics_match:
            return self._parse_metrics_line(metrics_match)

        # Epoch 표시 파싱
        epoch_match = self.epoch_pattern.search(line)
        if epoch_match:
            return self._parse_epoch_line(epoch_match)

        # 진행률 바 파싱
        progress_match = self.progress_pattern.search(line)
        if progress_match:
            return self._parse_progress_line(progress_match)

        return None

    def _parse_metrics_line(self, match: re.Match) -> Dict[str, Any]:
        """메트릭 라인 파싱"""
        try:
            groups = match.groups()

            self.current_metrics.epoch = int(groups[0])
            self.current_metrics.total_epochs = int(groups[1])
            self.current_metrics.gpu_mem = f"{groups[2]}G"
            self.current_metrics.box_loss = float(groups[3])
            self.current_metrics.obj_loss = float(groups[4])
            self.current_metrics.cls_loss = float(groups[5])
            self.current_metrics.total_loss = float(groups[6])
            self.current_metrics.instances = int(groups[7])
            self.current_metrics.img_size = int(groups[8])
            self.current_metrics.precision = float(groups[9])
            self.current_metrics.recall = float(groups[10])
            self.current_metrics.map50 = float(groups[11])
            self.current_metrics.map95 = float(groups[12])
            self.current_metrics.val_box_loss = float(groups[13])
            self.current_metrics.val_obj_loss = float(groups[14])
            self.current_metrics.val_cls_loss = float(groups[15])

            # 진행률 계산
            if self.current_metrics.total_epochs > 0:
                self.current_metrics.progress_percent = (
                    (self.current_metrics.epoch + 1) / self.current_metrics.total_epochs * 100
                )

            self.last_epoch = self.current_metrics.epoch

            return {
                'type': 'metrics',
                'data': self.current_metrics.to_dict()
            }

        except (ValueError, IndexError) as e:
            print(f"⚠️ 메트릭 파싱 실패: {e}")
            return None

    def _parse_epoch_line(self, match: re.Match) -> Dict[str, Any]:
        """Epoch 라인 파싱"""
        try:
            epoch = int(match.group(1))
            total_epochs = int(match.group(2))

            self.current_metrics.epoch = epoch
            self.current_metrics.total_epochs = total_epochs

            if total_epochs > 0:
                self.current_metrics.progress_percent = (epoch / total_epochs) * 100

            return {
                'type': 'epoch',
                'data': {
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'progress_percent': self.current_metrics.progress_percent
                }
            }

        except (ValueError, IndexError) as e:
            print(f"⚠️ Epoch 파싱 실패: {e}")
            return None

    def _parse_progress_line(self, match: re.Match) -> Dict[str, Any]:
        """진행률 라인 파싱"""
        try:
            progress = int(match.group(1))

            return {
                'type': 'progress',
                'data': {
                    'progress_percent': progress
                }
            }

        except (ValueError, IndexError) as e:
            print(f"⚠️ 진행률 파싱 실패: {e}")
            return None

    def get_current_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        return self.current_metrics.to_dict()

    def reset(self):
        """파서 리셋"""
        self.current_metrics = TrainingMetrics()
        self.last_epoch = 0


# 테스트 코드
if __name__ == "__main__":
    print("🧪 YOLOv7LogParser 테스트...")

    parser = YOLOv7LogParser()

    # 테스트 로그 라인들
    test_lines = [
        "     0/299     2.59G   0.02872  0.007841  0.009021   0.04558         8       640     0.884    0.5205    0.6525     0.519   0.03446   0.02027   0.01341",
        "     1/299     2.51G   0.02611    0.0059  0.004185    0.0362         3       640    0.6828    0.7982    0.8666    0.7177   0.03103   0.01723   0.01167",
        "Epoch 10/300",
        "100%|██████████| 15/15"
    ]

    for line in test_lines:
        result = parser.parse_line(line)
        if result:
            print(f"✅ 파싱 성공: {result['type']}")
            if result['type'] == 'metrics':
                data = result['data']
                print(f"   Epoch: {data['epoch']}/{data['total_epochs']}")
                print(f"   Precision: {data['precision']:.3f}, Recall: {data['recall']:.3f}")
                print(f"   mAP@0.5: {data['map50']:.3f}, mAP@0.5:0.95: {data['map95']:.3f}")
        else:
            print(f"❌ 파싱 실패: {line[:50]}")

    print("\n✅ 테스트 완료!")
