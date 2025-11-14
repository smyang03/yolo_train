import sys
import os
import torch
import cv2
import numpy as np
import time
from pathlib import Path
import torch.backends.cudnn as cudnn

# Qt 플랫폼 플러그인 오류 수정을 위한 환경 변수 설정
# QT_QPA_PLATFORM_PLUGIN_PATH 환경 변수를 PyQt5 plugins 디렉토리로 설정
if hasattr(sys, 'frozen'):
    # PyInstaller로 생성된 실행 파일인 경우
    os.environ['PATH'] = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt', 'plugins') + os.pathsep + os.environ['PATH']
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt', 'plugins', 'platforms')
else:
    # 소스에서 실행하는 경우
    import site
    for prefix_path in site.PREFIXES:
        qt_plugin_path = os.path.join(prefix_path, 'Lib', 'site-packages', 'PyQt5', 'Qt', 'plugins', 'platforms')
        if os.path.exists(qt_plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
            break

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QComboBox, QPushButton, QFileDialog, QSlider, 
                           QGroupBox, QLineEdit, QCheckBox, QMessageBox, QProgressBar,
                           QScrollArea, QFrame, QDialog, QListWidget, QTabWidget, QListView,QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QStringListModel
from PyQt5.QtGui import QPixmap, QImage
import numpy as np

# Import necessary functions from the original script
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel, time_synchronized
import os
import sys
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import psutil

def resource_path(relative_path):
    """PyInstaller에서 사용하기 위한 절대 경로 가져오기"""
    try:
        # PyInstaller가 생성한 임시 폴더
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

class AdaptiveDetectionThread(QThread):
    def __init__(self, config, parent=None):
        super().__init__()
        self.config = config
        self.running = True
        self.model_performances = {}
        self.parent = parent
        self.detection_results = {}
        
        # 동적 해상도 처리 설정
        self.resolution_manager = ResolutionManager()
        self.memory_threshold = 0.8  # 시스템 메모리의 80%
        self.frame_counter = 0
        self.memory_check_interval = 30
        
        # 이전 해상도 기억 (성능 최적화용)
        self.last_resolution = None
        self.last_grid_config = None
class ResolutionManager:
    """해상도 관리 및 최적화 클래스"""
    
    def __init__(self):
        # 해상도별 메모리 사용량 추정 테이블 (MB 단위)
        self.memory_estimates = {
            (1920, 1080): 6.2,  # Full HD
            (1280, 720): 2.8,   # HD
            (854, 480): 1.2,    # 480p
            (640, 360): 0.7,    # 360p
            (426, 240): 0.3,    # 240p
        }
        
        # 표준 해상도 목록 (성능 최적화용)
        self.standard_resolutions = [
            (3840, 2160),  # 4K
            (2560, 1440),  # 2K
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (854, 480),    # 480p
            (640, 360),    # 360p
            (426, 240),    # 240p
        ]
        
        # 최소/최대 해상도 제한
        self.min_resolution = (320, 240)
        self.max_resolution = (1920, 1080)  # 메모리 절약을 위한 최대 제한
        
    def estimate_memory_usage(self, width, height, num_models=1, dtype_size=1):
        """해상도별 예상 메모리 사용량 계산 (MB)"""
        # 기본 이미지 메모리 (3채널)
        base_memory = (width * height * 3 * dtype_size) / (1024 * 1024)
        
        # 모델별 복사본 + 처리 오버헤드
        total_memory = base_memory * (num_models + 2) * 1.5
        
        return total_memory
    
    def get_optimal_resolution(self, original_width, original_height, num_models=1, 
                             available_memory_mb=None):
        """최적 해상도 계산"""
        
        if available_memory_mb is None:
            memory_info = psutil.virtual_memory()
            available_memory_mb = (memory_info.available / (1024 * 1024)) * 0.3  # 30%만 사용
        
        # 원본 비율 계산
        aspect_ratio = original_width / original_height
        
        # 원본이 최대 제한 이하면 그대로 사용
        if (original_width <= self.max_resolution[0] and 
            original_height <= self.max_resolution[1]):
            
            estimated = self.estimate_memory_usage(original_width, original_height, num_models)
            if estimated <= available_memory_mb:
                return original_width, original_height
        
        # 메모리 제약에 맞는 최대 해상도 찾기
        for target_width, target_height in self.standard_resolutions:
            # 비율에 맞게 조정
            if target_width / target_height > aspect_ratio:
                # 세로가 더 긴 경우
                adjusted_width = int(target_height * aspect_ratio)
                adjusted_height = target_height
            else:
                # 가로가 더 긴 경우
                adjusted_width = target_width
                adjusted_height = int(target_width / aspect_ratio)
            
            # 최소 해상도 체크
            if (adjusted_width < self.min_resolution[0] or 
                adjusted_height < self.min_resolution[1]):
                continue
            
            # 메모리 사용량 체크
            estimated = self.estimate_memory_usage(adjusted_width, adjusted_height, num_models)
            if estimated <= available_memory_mb:
                return adjusted_width, adjusted_height
        
        # 모든 표준 해상도가 메모리 제약을 벗어나면 최소 해상도 반환
        return self.min_resolution
    
    def create_adaptive_grid(self, images, max_width=1920, max_height=1080):
        """이미지 목록에 대한 적응적 그리드 레이아웃 생성"""
        if not images:
            return None
        
        num_images = len(images)
        
        # 첫 번째 이미지의 크기 기준으로 그리드 계산
        img_height, img_width = images[0].shape[:2]
        
        # 그리드 배치 최적화
        grid_config = self.calculate_optimal_grid(num_images, img_width, img_height, 
                                                max_width, max_height)
        
        return self.create_grid_image(images, grid_config)
    
    def calculate_optimal_grid(self, num_images, img_width, img_height, max_width, max_height):
        """최적 그리드 배치 계산"""
        best_config = None
        best_efficiency = 0
        
        # 가능한 그리드 배치 시도
        for cols in range(1, num_images + 1):
            rows = math.ceil(num_images / cols)
            
            # 각 이미지의 크기 계산
            cell_width = max_width // cols
            cell_height = max_height // rows
            
            # 비율 유지하면서 셀에 맞는 크기 계산
            scale_w = cell_width / img_width
            scale_h = cell_height / img_height
            scale = min(scale_w, scale_h)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 실제 그리드 크기
            actual_width = cols * new_width
            actual_height = rows * new_height
            
            # 효율성 계산 (화면 사용률)
            if actual_width <= max_width and actual_height <= max_height:
                efficiency = (actual_width * actual_height) / (max_width * max_height)
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_config = {
                        'rows': rows,
                        'cols': cols,
                        'cell_width': new_width,
                        'cell_height': new_height,
                        'grid_width': actual_width,
                        'grid_height': actual_height
                    }
        
        # 기본 설정 (2열 고정)
        if best_config is None:
            cols = min(2, num_images)
            rows = math.ceil(num_images / cols)
            cell_width = max_width // cols
            cell_height = max_height // rows
            
            scale_w = cell_width / img_width
            scale_h = cell_height / img_height
            scale = min(scale_w, scale_h) * 0.8  # 여유 공간 확보
            
            best_config = {
                'rows': rows,
                'cols': cols,
                'cell_width': int(img_width * scale),
                'cell_height': int(img_height * scale),
                'grid_width': cols * int(img_width * scale),
                'grid_height': rows * int(img_height * scale)
            }
        
        return best_config
    
    def create_grid_image(self, images, grid_config):
        """그리드 설정에 따라 이미지 배치"""
        try:
            grid_image = np.zeros((grid_config['grid_height'], grid_config['grid_width'], 3), 
                                dtype=np.uint8)
            
            for idx, img in enumerate(images):
                if idx >= grid_config['rows'] * grid_config['cols']:
                    break
                
                # 그리드 위치 계산
                row = idx // grid_config['cols']
                col = idx % grid_config['cols']
                
                # 이미지 크기 조정
                resized_img = cv2.resize(img, 
                                       (grid_config['cell_width'], grid_config['cell_height']),
                                       interpolation=cv2.INTER_AREA)
                
                # 그리드에 배치
                y_start = row * grid_config['cell_height']
                x_start = col * grid_config['cell_width']
                y_end = y_start + grid_config['cell_height']
                x_end = x_start + grid_config['cell_width']
                
                grid_image[y_start:y_end, x_start:x_end] = resized_img
            
            return grid_image
            
        except Exception as e:
            print(f"그리드 이미지 생성 중 오류: {e}")
            # 실패시 첫 번째 이미지만 반환
            return images[0] if images else None
# 메모리 모니터링 유틸리티

class ResultsSummaryDialog(QDialog):
    def __init__(self, results_dir, parent=None):
        super().__init__(parent)
        self.results_dir = results_dir
        self.setWindowTitle("Detection Results Summary")
        self.setMinimumSize(800, 600)
        self.report_files = []
        self.summaries = {}
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 상단 설명
        header = QLabel("Summary of Detection Results")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        # 탭 위젯 (각 비교 유형별로 탭 분리)
        self.tabs = QTabWidget()
        
        # 요약 탭
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        # 요약 텍스트 영역
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        # 모델별 탭
        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)
        
        # 모델 선택 콤보박스
        model_selection = QHBoxLayout()
        model_selection.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        model_selection.addWidget(self.model_combo)
        models_layout.addLayout(model_selection)
        
        # 모델별 성능 텍스트 영역
        self.model_text = QTextEdit()
        self.model_text.setReadOnly(True)
        models_layout.addWidget(self.model_text)
        
        # 비디오별 탭
        videos_tab = QWidget()
        videos_layout = QVBoxLayout(videos_tab)
        
        # 비디오 선택 콤보박스
        video_selection = QHBoxLayout()
        video_selection.addWidget(QLabel("Select Video:"))
        self.video_combo = QComboBox()
        video_selection.addWidget(self.video_combo)
        videos_layout.addLayout(video_selection)
        
        # 비디오별 성능 텍스트 영역
        self.video_text = QTextEdit()
        self.video_text.setReadOnly(True)
        videos_layout.addWidget(self.video_text)
        
        # 클래스별 탭
        classes_tab = QWidget()
        classes_layout = QVBoxLayout(classes_tab)
        
        # 클래스 선택 콤보박스
        class_selection = QHBoxLayout()
        class_selection.addWidget(QLabel("Select Class:"))
        self.class_combo = QComboBox()
        class_selection.addWidget(self.class_combo)
        classes_layout.addLayout(class_selection)
        # 그래프 선택 버튼 그룹 추가

        graph_type_layout = QHBoxLayout()
        graph_type_layout.addWidget(QLabel("Graph Type:"))
        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(["Models Comparison", "Video Distribution", "Model-Video Heatmap"])
        self.graph_type_combo.currentIndexChanged.connect(self.update_class_graph)
        graph_type_layout.addWidget(self.graph_type_combo)
        classes_layout.addLayout(graph_type_layout)
            # 클래스별 그래프 영역 추가
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        classes_layout.addWidget(self.canvas)
        # 클래스별 성능 텍스트 영역
        self.class_text = QTextEdit()
        self.class_text.setReadOnly(True)
        classes_layout.addWidget(self.class_text)
        
        # 탭 추가
        self.tabs.addTab(summary_tab, "Overall Summary")
        self.tabs.addTab(models_tab, "Models Comparison")
        self.tabs.addTab(videos_tab, "Videos Comparison")
        self.tabs.addTab(classes_tab, "Classes Analysis")
        
        layout.addWidget(self.tabs)
        
        # 내보내기 버튼
        export_btn = QPushButton("Export Summary to TXT")
        export_btn.clicked.connect(self.export_summary)
        layout.addWidget(export_btn)
        
        self.setLayout(layout)
        
        # 콤보박스 변경 시 이벤트 연결
        self.model_combo.currentIndexChanged.connect(self.update_model_text)
        self.video_combo.currentIndexChanged.connect(self.update_video_text)
        self.class_combo.currentIndexChanged.connect(self.update_class_text)
        
        # 보고서 파일 로드 및 분석
        self.load_reports()
    def export_summary(self):
        """요약 결과를 텍스트 파일로 내보내기"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Summary Report", "", "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # 전체 요약
                    f.write(self.summary_text.toPlainText())
                    f.write("\n\n")
                    
                    # 모델별 분석 (현재 선택된 모델)
                    f.write("=" * 50 + "\n")
                    f.write(self.model_text.toPlainText())
                    f.write("\n\n")
                    
                    # 비디오별 분석 (현재 선택된 비디오)
                    f.write("=" * 50 + "\n")
                    f.write(self.video_text.toPlainText())
                    f.write("\n\n")
                    
                    # 클래스별 분석 (현재 선택된 클래스)
                    f.write("=" * 50 + "\n")
                    f.write(self.class_text.toPlainText())
                
                QMessageBox.information(self, "Export Successful", 
                                    f"Summary report exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", 
                                    f"Failed to save report: {str(e)}")
    def update_class_content(self):
        """클래스 탭 내용 업데이트 (텍스트와 그래프 모두)"""
        self.update_class_text()
        self.update_class_graph()

    def update_class_graph(self):
        """클래스 그래프 업데이트"""
        class_name = self.class_combo.currentText()
        if not class_name:
            return
        
        # 그래프 초기화
        self.figure.clear()
        
        graph_type = self.graph_type_combo.currentText()
        
        if graph_type == "Models Comparison":
            ax = self.figure.add_subplot(111)
            # 모델별 감지 수 수집
            model_data = {}
            for video_data in self.summaries.values():
                for model_name, model_data_item in video_data['models'].items():
                    if class_name in model_data_item['classes']:
                        if model_name not in model_data:
                            model_data[model_name] = 0
                        model_data[model_name] += model_data_item['classes'][class_name]
            
            # 그래프 생성 (막대 그래프)
            models = list(model_data.keys())
            counts = [model_data[model] for model in models]
            
            if not models:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            else:
                bars = ax.bar(models, counts, color='skyblue')
                ax.set_xlabel('Models')
                ax.set_ylabel('Detection Count')
                ax.set_title(f'Detection Count by Model for Class: {class_name}')
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                
                # 막대 위에 값 표시
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
        
        elif graph_type == "Video Distribution":
            # 모든 비디오 표시하기 위해 figure 레이아웃 조정
            # 비디오 수에 따라 그래프 크기 동적 조정
            video_data = {}
            for video_name, video_data_item in self.summaries.items():
                # 각 비디오에 대한 모델별 감지 수 저장
                video_data[video_name] = {'total': 0, 'models': {}}
                
                for model_name, model_data in video_data_item['models'].items():
                    if class_name in model_data['classes']:
                        count = model_data['classes'][class_name]
                        video_data[video_name]['models'][model_name] = count
                        video_data[video_name]['total'] += count
            
            # 내림차순 정렬
            sorted_videos = sorted(video_data.items(), key=lambda x: x[1]['total'], reverse=True)
            
            if not sorted_videos:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            else:
                # 비디오 수에 따라 그래프 높이 조정
                num_videos = len(sorted_videos)
                
                # 수직 스택 바 차트로 변경 (모델별로 색상 구분)
                ax = self.figure.add_subplot(111)
                
                videos = [v[0][:20] + ('...' if len(v[0]) > 20 else '') for v in sorted_videos]
                
                # 모든 모델 이름 수집
                all_models = set()
                for video_data in self.summaries.values():
                    for model_name in video_data['models'].keys():
                        all_models.add(model_name)
                
                # 모델 이름 정렬
                sorted_models = sorted(all_models)
                
                # 각 모델별 데이터 준비
                model_data_by_video = {model: [] for model in sorted_models}
                
                for video_name, stats in sorted_videos:
                    for model in sorted_models:
                        if model in stats['models']:
                            model_data_by_video[model].append(stats['models'][model])
                        else:
                            model_data_by_video[model].append(0)
                
                # 누적 막대그래프 그리기
                bottom = np.zeros(len(videos))
                bars = []
                
                # 각 모델에 대해 서로 다른 색상 사용
                colors = plt.cm.tab10.colors
                
                for i, model in enumerate(sorted_models):
                    if sum(model_data_by_video[model]) > 0:  # 데이터가 있는 모델만 표시
                        color = colors[i % len(colors)]
                        bar = ax.barh(videos, model_data_by_video[model], left=bottom, label=model, color=color)
                        bars.append(bar)
                        bottom += np.array(model_data_by_video[model])
                
                # y축 라벨 설정
                if len(videos) > 15:
                    # 비디오가 많으면 y축 라벨 간격 조정
                    y_label_step = max(1, len(videos) // 15)
                    ax.set_yticks(range(0, len(videos), y_label_step))
                    ax.set_yticklabels([videos[i] for i in range(0, len(videos), y_label_step)])
                else:
                    ax.set_yticks(range(len(videos)))
                    ax.set_yticklabels(videos)
                
                # 축 라벨 설정
                ax.set_xlabel('Detection Count')
                ax.set_title(f'All Videos by Detection Count for Class: {class_name}')
                
                # 그래프가 너무 크면 표시 영역 제한
                if len(videos) > 20:
                    plt.subplots_adjust(left=0.25)  # 왼쪽 여백 늘리기
                
                # 범례 추가
                ax.legend(loc='upper right')
                
                # 각 비디오의 총 감지 수 표시
                for i, (video_name, stats) in enumerate(sorted_videos):
                    ax.text(stats['total'] + 0.5, i, f"{stats['total']}", va='center')
        
        elif graph_type == "Model-Video Heatmap":
            ax = self.figure.add_subplot(111)
            # 히트맵 데이터 준비
            model_video_data = {}
            
            # 모든 모델과 비디오 목록 (중복 제거)
            all_models = set()
            all_videos = set()
            
            # 데이터 수집
            for video_name, video_data in self.summaries.items():
                for model_name, model_data in video_data['models'].items():
                    if class_name in model_data['classes']:
                        all_models.add(model_name)
                        all_videos.add(video_name)
                        
                        if model_name not in model_video_data:
                            model_video_data[model_name] = {}
                        
                        model_video_data[model_name][video_name] = model_data['classes'][class_name]
            
            # 비디오를 총 감지 수 기준 정렬
            video_total_counts = {}
            for video_name in all_videos:
                total = 0
                for model_data in model_video_data.values():
                    if video_name in model_data:
                        total += model_data[video_name]
                video_total_counts[video_name] = total
            
            sorted_videos = [v[0] for v in sorted(video_total_counts.items(), 
                                            key=lambda x: x[1], reverse=True)]
            
            # 모델 목록 정렬
            sorted_models = sorted(all_models)
            
            if not sorted_videos or not sorted_models:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            else:
                # 히트맵 높이 조절 (비디오 수가 많으면)
                if len(sorted_videos) > 20:
                    top_videos = sorted_videos[:20]  # 일단 상위 20개만 표시
                    ax.text(0.5, -0.1, f"Note: Showing top 20 out of {len(sorted_videos)} videos", 
                            ha='center', va='center', transform=ax.transAxes, fontsize=9)
                else:
                    top_videos = sorted_videos
                
                # 히트맵 데이터 행렬 생성
                data_matrix = np.zeros((len(sorted_models), len(top_videos)))
                
                for i, model in enumerate(sorted_models):
                    for j, video in enumerate(top_videos):
                        if model in model_video_data and video in model_video_data[model]:
                            data_matrix[i, j] = model_video_data[model][video]
                
                # 히트맵 생성
                im = ax.imshow(data_matrix, cmap='YlGnBu')
                
                # 축 라벨 설정
                ax.set_xticks(range(len(top_videos)))
                ax.set_yticks(range(len(sorted_models)))
                
                # 비디오 이름이 길면 잘라서 표시
                video_labels = [v[:15] + '...' if len(v) > 15 else v for v in top_videos]
                ax.set_xticklabels(video_labels, rotation=45, ha='right')
                ax.set_yticklabels(sorted_models)
                
                ax.set_xlabel('Videos')
                ax.set_ylabel('Models')
                ax.set_title(f'Detection Heatmap for Class: {class_name}')
                
                # 컬러바 추가
                cbar = self.figure.colorbar(im)
                cbar.set_label('Detection Count')
                
                # 셀에 값 표시
                for i in range(len(sorted_models)):
                    for j in range(len(top_videos)):
                        if data_matrix[i, j] > 0:
                            text_color = 'white' if data_matrix[i, j] > data_matrix.max() / 2 else 'black'
                            ax.text(j, i, int(data_matrix[i, j]), ha='center', va='center', color=text_color)
        
        self.figure.tight_layout()
        self.canvas.draw()
    def load_reports(self):
        """모든 performance_report.txt 파일 로드 및 분석"""
        try:
            # 결과 디렉토리 내 모든 하위 디렉토리 검색
            all_subdirs = [f.path for f in os.scandir(self.results_dir) if f.is_dir()]
            
            # 각 하위 디렉토리에서 performance_report.txt 찾기
            for subdir in all_subdirs:
                report_path = os.path.join(subdir, 'performance_report.txt')
                if os.path.exists(report_path):
                    self.report_files.append(report_path)
                    
                    # 비디오 이름 (폴더 이름)
                    video_name = os.path.basename(subdir)
                    
                    # 보고서 파싱
                    self.parse_report(report_path, video_name)
            
            # 콤보박스 업데이트
            self.update_combos()
            
            # 요약 업데이트
            self.update_summary()
            
        except Exception as e:
            self.summary_text.setText(f"Error loading reports: {str(e)}")
    
    def parse_report(self, report_path, video_name):
        """단일 보고서 파일 파싱"""
        try:
            with open(report_path, 'r') as f:
                report_text = f.read()
                
            # 보고서 저장
            self.summaries[video_name] = {
                'text': report_text,
                'models': {},
                'classes': {}
            }
            
            # 모델 정보 추출
            current_model = None
            in_class_section = False
            
            for line in report_text.split('\n'):
                line = line.strip()
                
                # 모델 시작
                if line.startswith("Model:"):
                    current_model = line.split("Model:")[1].strip()
                    self.summaries[video_name]['models'][current_model] = {
                        'fps': 0,
                        'actual_fps': 0,
                        'inference_time': 0,
                        'detections': 0,
                        'classes': {}
                    }
                
                # 모델 FPS
                elif current_model and "Inference FPS:" in line:
                    fps = float(line.split("Inference FPS:")[1].split()[0])
                    self.summaries[video_name]['models'][current_model]['fps'] = fps
                
                # 실제 FPS
                elif current_model and "Actual FPS:" in line:
                    actual_fps = float(line.split("Actual FPS:")[1].split()[0])
                    self.summaries[video_name]['models'][current_model]['actual_fps'] = actual_fps
                
                # 추론 시간
                elif current_model and "Average inference time:" in line:
                    time_ms = float(line.split("Average inference time:")[1].split()[0])
                    self.summaries[video_name]['models'][current_model]['inference_time'] = time_ms
                
                # 감지 수
                elif current_model and "Total detected objects:" in line:
                    detections = int(line.split("Total detected objects:")[1].strip())
                    self.summaries[video_name]['models'][current_model]['detections'] = detections
                
                # 클래스 섹션 시작
                elif current_model and "Class Frequency:" in line:
                    in_class_section = True
                
                # 클래스 빈도
                elif current_model and in_class_section and ":" in line and not line.startswith("Model:"):
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        class_name = parts[0].strip()
                        count = int(parts[1].strip())
                        
                        # 모델별 클래스 빈도
                        self.summaries[video_name]['models'][current_model]['classes'][class_name] = count
                        
                        # 비디오별 클래스 빈도 (모든 모델 합산)
                        if class_name not in self.summaries[video_name]['classes']:
                            self.summaries[video_name]['classes'][class_name] = 0
                        self.summaries[video_name]['classes'][class_name] += count
                
                # 모델 섹션 종료
                elif in_class_section and line == "":
                    in_class_section = False
        
        except Exception as e:
            print(f"Error parsing report {report_path}: {str(e)}")
    
    def update_combos(self):
        """콤보박스 업데이트"""
        # 모델 목록 (중복 제거)
        all_models = set()
        for video_data in self.summaries.values():
            for model in video_data['models'].keys():
                all_models.add(model)
        
        self.model_combo.clear()
        self.model_combo.addItems(sorted(all_models))
        
        # 비디오 목록
        self.video_combo.clear()
        self.video_combo.addItems(sorted(self.summaries.keys()))
        
        # 클래스 목록 (중복 제거)
        all_classes = set()
        for video_data in self.summaries.values():
            for model_data in video_data['models'].values():
                for class_name in model_data['classes'].keys():
                    all_classes.add(class_name)
        
        self.class_combo.clear()
        self.class_combo.addItems(sorted(all_classes))
    
    def update_summary(self):
        """전체 요약 업데이트"""
        if not self.summaries:
            self.summary_text.setText("No reports found.")
            return
        
        summary = "=== Detection Results Summary ===\n\n"
        
        # 처리된 비디오 수
        summary += f"Total Videos Processed: {len(self.summaries)}\n"
        
        # 모델 수
        all_models = set()
        for video_data in self.summaries.values():
            for model in video_data['models'].keys():
                all_models.add(model)
        
        summary += f"Models Used: {len(all_models)}\n\n"
        
        # 모델별 통계
        summary += "Model Performance (Averaged across all videos):\n"
        model_stats = {}
        
        for model in all_models:
            model_stats[model] = {
                'fps': [],
                'actual_fps': [],
                'inference_time': [],
                'detections': 0,
                'videos': 0
            }
        
        # 모델별 데이터 수집
        for video_name, video_data in self.summaries.items():
            for model_name, model_data in video_data['models'].items():
                model_stats[model_name]['fps'].append(model_data['fps'])
                model_stats[model_name]['actual_fps'].append(model_data['actual_fps'])
                model_stats[model_name]['inference_time'].append(model_data['inference_time'])
                model_stats[model_name]['detections'] += model_data['detections']
                model_stats[model_name]['videos'] += 1
        
        # 모델별 평균 계산 및 표시
        for model_name, stats in model_stats.items():
            avg_fps = sum(stats['fps']) / len(stats['fps']) if stats['fps'] else 0
            avg_actual_fps = sum(stats['actual_fps']) / len(stats['actual_fps']) if stats['actual_fps'] else 0
            avg_time = sum(stats['inference_time']) / len(stats['inference_time']) if stats['inference_time'] else 0
            
            summary += f"\n{model_name}:\n"
            summary += f"  - Average Inference FPS: {avg_fps:.2f}\n"
            summary += f"  - Average Actual FPS: {avg_actual_fps:.2f}\n"
            summary += f"  - Average Inference Time: {avg_time:.2f} ms\n"
            summary += f"  - Total Detections: {stats['detections']}\n"
            summary += f"  - Processed Videos: {stats['videos']}\n"
        
        # 클래스별 통계
        summary += "\nClass Detection Statistics (All models, all videos):\n"
        
        class_stats = {}
        for video_data in self.summaries.values():
            for model_data in video_data['models'].values():
                for class_name, count in model_data['classes'].items():
                    if class_name not in class_stats:
                        class_stats[class_name] = 0
                    class_stats[class_name] += count
        
        # 클래스별 감지 수 내림차순 정렬
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes:
            summary += f"  - {class_name}: {count} detections\n"
        
        self.summary_text.setText(summary)
    
    def update_model_text(self):
        """모델별 비교 업데이트"""
        model_name = self.model_combo.currentText()
        if not model_name:
            return
        
        text = f"=== Performance for Model: {model_name} ===\n\n"
        
        # 비디오별 모델 성능
        text += "Performance across videos:\n\n"
        
        model_data = []
        for video_name, video_data in self.summaries.items():
            if model_name in video_data['models']:
                model_info = video_data['models'][model_name]
                model_data.append({
                    'video': video_name,
                    'fps': model_info['fps'],
                    'actual_fps': model_info['actual_fps'],
                    'inference_time': model_info['inference_time'],
                    'detections': model_info['detections']
                })
        
        # 감지 수 기준 내림차순 정렬
        model_data.sort(key=lambda x: x['detections'], reverse=True)
        
        for data in model_data:
            text += f"Video: {data['video']}\n"
            text += f"  - Inference FPS: {data['fps']:.2f}\n"
            text += f"  - Actual FPS: {data['actual_fps']:.2f}\n"
            text += f"  - Inference Time: {data['inference_time']:.2f} ms\n"
            text += f"  - Detections: {data['detections']}\n\n"
        
        # 클래스별 감지 통계
        text += "Class detection statistics:\n\n"
        
        class_stats = {}
        for video_data in self.summaries.values():
            if model_name in video_data['models']:
                for class_name, count in video_data['models'][model_name]['classes'].items():
                    if class_name not in class_stats:
                        class_stats[class_name] = 0
                    class_stats[class_name] += count
        
        # 감지 수 기준 내림차순 정렬
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes:
            text += f"  - {class_name}: {count} detections\n"
        
        self.model_text.setText(text)
    
    def update_video_text(self):
        """비디오별 비교 업데이트"""
        video_name = self.video_combo.currentText()
        if not video_name or video_name not in self.summaries:
            return
        
        video_data = self.summaries[video_name]
        text = f"=== Performance for Video: {video_name} ===\n\n"
        
        # 모델별 성능
        text += "Model Performance:\n\n"
        
        # 감지 수 기준 내림차순 정렬
        sorted_models = sorted(video_data['models'].items(), 
                              key=lambda x: x[1]['detections'], reverse=True)
        
        for model_name, model_info in sorted_models:
            text += f"Model: {model_name}\n"
            text += f"  - Inference FPS: {model_info['fps']:.2f}\n"
            text += f"  - Actual FPS: {model_info['actual_fps']:.2f}\n"
            text += f"  - Inference Time: {model_info['inference_time']:.2f} ms\n"
            text += f"  - Detections: {model_info['detections']}\n\n"
        
        # 클래스별 감지 통계
        text += "Class detection statistics (all models):\n\n"
        
        # 감지 수 기준 내림차순 정렬
        sorted_classes = sorted(video_data['classes'].items(), 
                               key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes:
            text += f"  - {class_name}: {count} detections\n"
        
        self.video_text.setText(text)
    
    def update_class_text(self):
        """클래스별 분석 업데이트"""
        class_name = self.class_combo.currentText()
        if not class_name:
            return
        
        text = f"=== Analysis for Class: {class_name} ===\n\n"
        
        # 모델별 클래스 감지 통계
        text += "Detection by Model:\n\n"
        
        model_class_stats = {}
        for video_data in self.summaries.values():
            for model_name, model_data in video_data['models'].items():
                if class_name in model_data['classes']:
                    if model_name not in model_class_stats:
                        model_class_stats[model_name] = 0
                    model_class_stats[model_name] += model_data['classes'][class_name]
        
        # 감지 수 기준 내림차순 정렬
        sorted_models = sorted(model_class_stats.items(), 
                            key=lambda x: x[1], reverse=True)
        
        for model_name, count in sorted_models:
            text += f"  - {model_name}: {count} detections\n"
        
        # 비디오별 클래스 감지 통계 (개선된 부분)
        text += "\nDetection by Video:\n\n"
        
        # 비디오별로 각 모델의 감지 수 저장
        video_model_stats = {}
        for video_name, video_data in self.summaries.items():
            if video_name not in video_model_stats:
                video_model_stats[video_name] = {'total': 0, 'models': {}}
                
            for model_name, model_data in video_data['models'].items():
                if class_name in model_data['classes']:
                    count = model_data['classes'][class_name]
                    video_model_stats[video_name]['models'][model_name] = count
                    video_model_stats[video_name]['total'] += count
        
        # 총 감지 수 기준 내림차순 정렬
        sorted_videos = sorted(video_model_stats.items(), 
                            key=lambda x: x[1]['total'], reverse=True)
        
        for video_name, stats in sorted_videos:
            # 각 비디오에 대한 전체 감지 수
            text += f"  - {video_name}: {stats['total']} total detections\n"
            
            # 각 비디오에 대한 모델별 감지 수
            for model_name, count in sorted(stats['models'].items(), key=lambda x: x[1], reverse=True):
                text += f"      * {model_name}: {count} detections\n"
        
        self.class_text.setText(text)

# 추가: ComparisonViewerDialog 클래스 전체
class ComparisonViewerDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Model Comparison Viewer")
        self.setMinimumSize(800, 600)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # 비교 모드 선택
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Comparison Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Side by Side", "Overlay", "Difference"])
        self.mode_combo.currentIndexChanged.connect(self.update_comparison)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)
        
        # 모델 선택
        models_layout = QHBoxLayout()
        models_layout.addWidget(QLabel("Base Model:"))
        self.base_model_combo = QComboBox()
        self.compare_model_combo = QComboBox()
        
        # 모델 목록 추가
        model_names = list(self.results.keys())
        self.base_model_combo.addItems(model_names)
        self.compare_model_combo.addItems(model_names)
        if len(model_names) > 1:
            self.compare_model_combo.setCurrentIndex(1)
        
        self.base_model_combo.currentIndexChanged.connect(self.update_comparison)
        self.compare_model_combo.currentIndexChanged.connect(self.update_comparison)
        
        models_layout.addWidget(self.base_model_combo)
        models_layout.addWidget(QLabel("Compare With:"))
        models_layout.addWidget(self.compare_model_combo)
        layout.addLayout(models_layout)
        
        # 프레임 선택 슬라이더 (비디오의 경우)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        
        # 프레임 수 설정
        valid_frames = {}
        for model_name, frames in self.results.items():
            valid_frames[model_name] = sum(1 for f in frames if f is not None)

        self.max_frames = max(valid_frames.values()) if valid_frames else 0
        self.frame_slider.setMaximum(max(0, self.max_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.update_comparison)
                
        self.frame_label = QLabel("0")
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)
        layout.addLayout(slider_layout)
        
        # 비교 이미지 표시 영역
        self.comparison_view = QLabel()
        self.comparison_view.setAlignment(Qt.AlignCenter)
        self.comparison_view.setStyleSheet("background-color: black;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.comparison_view)
        layout.addWidget(scroll)
        
        # 통계 표시 영역
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)
        
        # 스냅샷 버튼
        snapshot_btn = QPushButton("Save Snapshot")
        snapshot_btn.clicked.connect(self.save_snapshot)
        layout.addWidget(snapshot_btn)
        
        self.setLayout(layout)
        
        # 초기 비교 업데이트
        self.update_comparison()
    
    def update_comparison(self):
        base_model = self.base_model_combo.currentText()
        compare_model = self.compare_model_combo.currentText()
        comparison_mode = self.mode_combo.currentText()
        frame_idx = self.frame_slider.value()
        
        # 프레임 라벨 업데이트
        self.frame_label.setText(str(frame_idx))
        
    # 모델이 결과에 없는 경우 확인
        if base_model not in self.results or compare_model not in self.results:
            self.comparison_view.setText(f"Model not found in results")
            return
        
        # 프레임 인덱스가 범위를 벗어나는 경우 확인
        if frame_idx >= len(self.results[base_model]) or frame_idx >= len(self.results[compare_model]):
            self.comparison_view.setText(f"Frame index out of range")
            return
        
        # None 값 확인 (중요)
        if self.results[base_model][frame_idx] is None or self.results[compare_model][frame_idx] is None:
            self.comparison_view.setText(f"Frame data is not available for this frame")
            return
            
        # 이미지 로드
        base_img = self.results[base_model][frame_idx].copy()
        compare_img = self.results[compare_model][frame_idx].copy()
        
        # 이미지 크기 확인 및 조정
        if base_img.shape != compare_img.shape:
            compare_img = cv2.resize(compare_img, (base_img.shape[1], base_img.shape[0]))
        
        # 비교 모드에 따른 처리
        if comparison_mode == "Side by Side":
            # 두 이미지를.가로로 합치기
            comparison = np.hstack((base_img, compare_img))
            
            # 모델 이름 표시
            cv2.putText(comparison, base_model, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, compare_model, (base_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        elif comparison_mode == "Overlay":
            # 두 이미지 알파 블렌딩 (50:50)
            comparison = cv2.addWeighted(base_img, 0.5, compare_img, 0.5, 0)
            
            # 모델 이름 표시
            cv2.putText(comparison, f"{base_model} + {compare_model}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        elif comparison_mode == "Difference":
            # 이미지 차이 계산
            comparison = cv2.absdiff(base_img, compare_img)
            
            # 차이를 더 잘 보이게 강조
            comparison = cv2.convertScaleAbs(comparison, alpha=2.0, beta=0)
            
            # 모델 이름 표시
            cv2.putText(comparison, f"{base_model} - {compare_model}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 통계 계산 및 표시
        base_boxes = self.count_detection_boxes(base_img)
        compare_boxes = self.count_detection_boxes(compare_img)
        
        stats_text = f"Base Model ({base_model}): {base_boxes} detections\n"
        stats_text += f"Compare Model ({compare_model}): {compare_boxes} detections\n"
        stats_text += f"Difference: {abs(base_boxes - compare_boxes)} detections"
        
        self.stats_label.setText(stats_text)
        
        # OpenCV 이미지를 QPixmap으로 변환
        comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        h, w, ch = comparison_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(comparison_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # QLabel에 이미지 표시
        self.comparison_view.setPixmap(QPixmap.fromImage(q_image))
    
    def count_detection_boxes(self, image):
        """이미지에서 경계 상자 수를 추정합니다 (간단한 근사치)"""
        # OpenCV의 Canny 엣지 검출 및 윤곽선 감지 사용
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 적당한 크기의 윤곽선만 카운트 (노이즈 제거)
        min_area = 100
        boxes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                boxes += 1
        
        return boxes
    
    def save_snapshot(self):
        """현재 비교 뷰를 이미지 파일로 저장합니다"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison Snapshot", "", "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            pixmap = self.comparison_view.pixmap()
            if pixmap and not pixmap.isNull():
                pixmap.save(file_path)
                QMessageBox.information(self, "Snapshot Saved", f"Comparison snapshot saved to {file_path}")

class ClassFilterDialog(QDialog):
    def __init__(self, model_path, current_filters=None, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.current_filters = current_filters or []
        self.setWindowTitle(f"Class Filter - {os.path.basename(model_path)}")
        self.setMinimumSize(400, 500)
        self.class_names = []
        self.checkboxes = []
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()

        # 헤더 추가
        header = QLabel(f"Select classes to detect:")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)
        
        # 모든 클래스 선택/해제 버튼
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(clear_all_btn)
        layout.addLayout(btn_layout)
        
        # 스크롤 영역 추가
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        
        # 모델 로드 시도 및 클래스 정보 추출
        try:
            model = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # 모델 형식에 따라 클래스 이름 추출
            if isinstance(model, dict):
                if 'model' in model:
                    if hasattr(model['model'], 'names'):
                        self.class_names = model['model'].names
                    elif 'names' in model:
                        self.class_names = model['names']
                    else:
                        self.class_names = model['model'].module.names if hasattr(model['model'], 'module') else []
                else:
                    # 모델에서 names 키 찾기
                    self.class_names = None
                    for key in model:
                        if 'names' in str(key).lower():
                            self.class_names = model[key]
                            break
            else:
                # 직접 모델인 경우
                self.class_names = model.names if hasattr(model, 'names') else []
            
            # 클래스 체크박스 추가
            if self.class_names:
                for i, name in enumerate(self.class_names):
                    checkbox = QCheckBox(f"{i}: {name}")
                    # 현재 필터에 있는 클래스인 경우 체크
                    checkbox.setChecked(i in self.current_filters)
                    self.checkboxes.append(checkbox)
                    self.scroll_layout.addWidget(checkbox)
            else:
                self.scroll_layout.addWidget(QLabel("Could not extract class information from this model."))
        except Exception as e:
            self.scroll_layout.addWidget(QLabel(f"Error loading model: {str(e)}"))
            error_details = QLabel(f"Details: {str(e)}")
            error_details.setWordWrap(True)
            self.scroll_layout.addWidget(error_details)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # 저장 & 취소 버튼
        buttons = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        
        # 필터 저장 기능 추가
        save_filter_btn = QPushButton("Save Filter to File")
        save_filter_btn.clicked.connect(self.save_filter_to_file)
        layout.addWidget(save_filter_btn)
        
        self.setLayout(layout)
    
    def select_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
    
    def clear_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
    
    def get_filtered_classes(self):
        filtered = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                filtered.append(i)
        return filtered
    
    def save_filter_to_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Filter Configuration", "", "Text Files (*.txt)"
        )
        if file_path:
            with open(file_path, 'w') as f:
                for i, checkbox in enumerate(self.checkboxes):
                    if checkbox.isChecked():
                        class_name = self.class_names[i]
                        f.write(f"{i}: {class_name}\n")
            QMessageBox.information(self, "Filter Saved", f"Filter configuration saved to {file_path}")

class ModelClassInfo(QDialog):
    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.setWindowTitle(f"Model Classes - {os.path.basename(model_path)}")
        self.setMinimumSize(400, 500)
        self.class_names = []  # 클래스 이름 저장 변수 추가
        self.initUI()
            
    def initUI(self):
        layout = QVBoxLayout()
        
        # Add header
        header = QLabel(f"Classes in model: {os.path.basename(self.model_path)}")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)
        
        # Add info text
        info = QLabel("Loading model to extract class information...")
        layout.addWidget(info)
        
        # Scroll area for class list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Load model and extract class info
        try:
            # Load model with torch.hub to avoid loading to GPU
            model = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Extract class names different ways depending on model format
            if isinstance(model, dict):
                # PT file saved with state_dict
                if 'model' in model:
                    if hasattr(model['model'], 'names'):
                        class_names = model['model'].names
                    elif 'names' in model:
                        class_names = model['names']
                    else:
                        class_names = model['model'].module.names if hasattr(model['model'], 'module') else None
                else:
                    # Try to find names in model
                    class_names = None
                    for key in model:
                        if 'names' in str(key).lower():
                            class_names = model[key]
                            break
            else:
                # Direct model
                class_names = model.names if hasattr(model, 'names') else None
            
            # Display classes
            if class_names:
                info.setText(f"Found {len(class_names)} classes:")
                
                # 클래스 이름 저장
                self.class_names = class_names
                
                # 클래스 목록 생성
                class_list = QListWidget()
                for i, name in enumerate(class_names):
                    class_list.addItem(f"{i}: {name}")
                
                scroll_layout.addWidget(class_list)
            else:
                info.setText("Could not extract class information from this model.")
        except Exception as e:
            info.setText(f"Error loading model: {str(e)}")
            error_details = QLabel(f"Details: {str(e)}")
            error_details.setWordWrap(True)
            scroll_layout.addWidget(error_details)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Add close button
        button_layout = QHBoxLayout()

        # 저장 버튼 추가
        save_btn = QPushButton("Save to TXT")
        save_btn.clicked.connect(self.save_to_txt)
        button_layout.addWidget(save_btn)

        # 닫기 버튼
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def save_to_txt(self):
        """클래스 정보를 txt 파일로 저장합니다"""
        if not self.class_names:
            QMessageBox.warning(self, "Error", "No class information available to save.")
            return
            
        # 저장 경로 선택
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Classes to TXT", "", "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # 헤더 정보 추가
                    f.write(f"# Class information for {os.path.basename(self.model_path)}\n")
                    f.write(f"# Total classes: {len(self.class_names)}\n")
                    f.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # 클래스 정보 기록
                    for i, name in enumerate(self.class_names):
                        f.write(f"{i}: {name}\n")
                
                QMessageBox.information(self, "Success", f"Class information saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

class OptimizedDetectionThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    update_progress = pyqtSignal(int)
    detection_finished = pyqtSignal(str)
    update_performance = pyqtSignal(dict)
    update_analysis_data = pyqtSignal(list, list, float)
    video_finished = pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__()
        self.config = config
        self.running = True
        self.model_performances = {}
        self.parent = parent
        self.detection_results = {}
        
        # 동적 해상도 처리 설정
        self.resolution_manager = ResolutionManager()
        self.memory_threshold = 0.75  # 시스템 메모리의 75%
        self.frame_counter = 0
        self.memory_check_interval = 20  # 20프레임마다 메모리 체크
        
        # FPS 제어 관련 변수
        self.target_fps = config.get('target_fps', 0)
        self.frame_time = 0 if self.target_fps <= 0 else 1.0 / self.target_fps
        self.last_frame_time = 0
        
        # 샘플 캡처 관련 변수
        self.enable_samples = config.get('enable_samples', False)
        self.max_samples = config.get('max_samples', 10)
        self.target_class = config.get('target_class', 'Any Class')
        self.sample_model = config.get('sample_model', 'Any Model')
        self.captured_samples = 0
        self.target_class_id = None
        
        if self.target_class and self.target_class != 'Any Class':
            try:
                parts = self.target_class.split(':', 1)
                if len(parts) > 0:
                    self.target_class_id = int(parts[0].strip())
            except ValueError:
                self.target_class_id = None
        
        self.sample_model_id = None
        if self.sample_model and self.sample_model != 'Any Model':
            try:
                parts = self.sample_model.split(':', 1)[0].strip()
                model_id = int(parts.replace('Model', '').strip()) - 1
                self.sample_model_id = model_id
            except ValueError:
                self.sample_model_id = None

    def check_memory_usage(self):
        """메모리 사용량 체크 및 정리"""
        try:
            memory_info = psutil.virtual_memory()
            if memory_info.percent > self.memory_threshold * 100:
                print(f"⚠️ 메모리 사용량이 높습니다: {memory_info.percent:.1f}%")
                # 가비지 컬렉션 강제 실행
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            return True
        except Exception as e:
            print(f"메모리 체크 중 오류: {e}")
            return True

    def safe_image_copy(self, image):
        """메모리 안전 이미지 복사"""
        try:
            # 현재 해상도 확인
            h, w = image.shape[:2]
            current_resolution = (w, h)
            
            # 메모리 사용량 체크
            if not self.check_memory_usage():
                # 메모리 부족시 해상도 축소
                optimal_w, optimal_h = self.resolution_manager.get_optimal_resolution(
                    w, h, num_models=len([m for m in self.config['models'] if m])
                )
                
                if optimal_w != w or optimal_h != h:
                    print(f"📏 메모리 절약을 위해 해상도 조정: {w}x{h} -> {optimal_w}x{optimal_h}")
                    image = cv2.resize(image, (optimal_w, optimal_h), interpolation=cv2.INTER_AREA)
            
            return image.copy()
            
        except np.core._exceptions._ArrayMemoryError:
            # 극단적 메모리 부족시 최소 해상도로 축소
            print("🚨 심각한 메모리 부족으로 최소 해상도로 조정")
            min_w, min_h = self.resolution_manager.min_resolution
            resized = cv2.resize(image, (min_w, min_h), interpolation=cv2.INTER_AREA)
            return resized.copy()
            
        except Exception as e:
            print(f"이미지 복사 중 오류: {e}")
            return image

    def process_detections_adaptive(self, all_preds, im0s_copy, model_names, names, colors, 
                                  save_img, view_img, frame):
        """동적 해상도에 맞는 검출 결과 처리"""
        
        processed_images = []
        original_height, original_width = im0s_copy.shape[:2]
        
        # 메모리 사용량에 따른 처리 방식 결정
        memory_info = psutil.virtual_memory()
        use_lightweight_mode = memory_info.percent > 80
        
        if use_lightweight_mode:
            print(f"💡 라이트 모드 활성화 (메모리 사용량: {memory_info.percent:.1f}%)")
        
        for j, (det, model_name) in enumerate(zip(all_preds, model_names)):
            try:
                # 이미지 복사 (메모리 최적화 적용)
                im0 = self.safe_image_copy(im0s_copy)
                
                if len(det):
                    # 바운딩 박스 그리기
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or view_img:
                            cls_idx = int(cls)
                            if 0 <= cls_idx < len(names):
                                label = f'{names[cls_idx]} {conf:.2f}'
                            else:
                                label = f'class{cls_idx} {conf:.2f}'
                            
                            # 라이트 모드에서는 더 얇은 선 사용
                            line_thickness = 1 if use_lightweight_mode else 2
                            plot_one_box(xyxy, im0, label=label, 
                                       color=colors[min(cls_idx, len(colors)-1)], 
                                       line_thickness=line_thickness)
                
                # 성능 정보 표시 (라이트 모드에서는 간소화)
                if not use_lightweight_mode:
                    self.add_performance_overlay(im0, model_name, j)
                else:
                    # 간단한 모델명만 표시
                    cv2.putText(im0, f"M{j+1}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                processed_images.append(im0)
                
                # 결과 저장
                if hasattr(self.parent, 'store_detection_result'):
                    self.parent.store_detection_result(frame, model_name, im0)
                
            except Exception as e:
                print(f"모델 {model_name} 처리 중 오류: {e}")
                continue
        
        return processed_images

    def add_performance_overlay(self, image, model_name, model_index):
        """성능 정보 오버레이 추가"""
        try:
            perf = self.model_performances.get(model_name, {})
            
            # 기본 성능 정보
            infer_fps = perf.get('fps', 0)
            actual_fps = perf.get('actual_fps', 0)
            
            # 메인 성능 텍스트
            main_text = f"Model {model_index+1}: {model_name}"
            fps_text = f"FPS: {infer_fps:.1f} | Actual: {actual_fps:.1f}"
            
            # 텍스트 위치 계산
            text_y = 40 + (model_index * 60)
            
            cv2.putText(image, main_text, (20, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, fps_text, (20, text_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 클래스별 감지 수 (상위 3개만)
            class_counts = perf.get('class_counts', {})
            if class_counts:
                y_offset = text_y + 40
                for idx, (cls_name, count) in enumerate(
                    sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                ):
                    cv2.putText(image, f"{cls_name}: {count}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 15
                    
        except Exception as e:
            print(f"성능 오버레이 추가 중 오류: {e}")

    def run(self):
        # Extract configuration
        models = self.config['models']
        source = self.config['source']
        img_size = self.config['img_size']
        conf_thres = self.config['conf_thres']
        iou_thres = self.config['iou_thres']
        device = self.config['device']
        save_path = self.config['save_path']
        view_img = self.config['view_img']
        save_img = self.config['save_img']
        
        # 메모리 정보 출력
        memory_info = get_memory_usage_info()
        print(f"🖥️  시스템 메모리: {memory_info['total_gb']:.1f}GB "
              f"(사용 가능: {memory_info['available_gb']:.1f}GB, "
              f"사용률: {memory_info['percent']:.1f}%)")
        
        # 다중 비디오 처리 여부 확인
        is_directory = self.config.get('is_directory', False)
        video_files = self.config.get('video_files', [])
        
        # Get the number of active models
        active_models = [m for m in models if m]
        num_active_models = len(active_models)
        
        if num_active_models == 0:
            self.detection_finished.emit("No models selected")
            return
        
        print(f"🤖 활성화된 모델 수: {num_active_models}")
        
        # Load models
        device = select_device(device)
        half = device.type != 'cpu'
        
        loaded_models = []
        model_names = []
        model_filters = {}

        for i, model_path in enumerate(active_models):
            if model_path:
                try:
                    if isinstance(model_path, dict):
                        model_filters[i] = model_path.get('filtered_classes', [])
                        path = model_path['path']
                    else:
                        path = model_path
                        model_filters[i] = []
                        
                    if not os.path.isabs(path):
                        actual_path = resource_path(path)
                    else:
                        actual_path = path
                        
                    print(f"📁 모델 로딩 중: {os.path.basename(actual_path)}")
                    model = attempt_load(actual_path, map_location=device)
                    model_name = Path(path).stem
                    if half:
                        model.half()
                    loaded_models.append(model)
                    model_names.append(model_name)
                    print(f"✅ 모델 로딩 완료: {model_name}")
                except Exception as e:
                    print(f"❌ 모델 로드 중 오류 발생: {path if isinstance(model_path, dict) else model_path}")
                    print(f"   오류 메시지: {str(e)}")
        
        self.model_performances = {
            model_name: {
                'inference_times': [],
                'fps': 0,
                'avg_inference_time': 0,
                'detections_count': 0,
                'class_counts': {}
            } for model_name in model_names
        }
                
        # Get stride from first model
        stride = int(loaded_models[0].stride.max())
        img_size = check_img_size(img_size, s=stride)
        
        # 폴더 내 모든 비디오를 처리하는 경우
        if is_directory:
            print(f"📁 디렉토리 모드: {len(video_files)}개 비디오 처리")
            for video_idx, video_file in enumerate(video_files):
                if not self.running:
                    break
                
                video_name = os.path.basename(video_file)
                print(f"🎬 처리 중: {video_name} ({video_idx+1}/{len(video_files)})")
                
                video_save_dir = Path(increment_path(Path(save_path) / video_name, exist_ok=True))
                (video_save_dir / 'labels').mkdir(parents=True, exist_ok=True)
                
                self.update_progress.emit(int((video_idx / len(video_files)) * 100))
                for model_name in self.model_performances:
                    self.model_performances[model_name]['class_counts'] = {}
                
                self.process_single_source(
                    video_file, loaded_models, model_names, model_filters,
                    img_size, stride, conf_thres, iou_thres, device, half,
                    video_save_dir, view_img, save_img, num_active_models
                )
                
                self.video_finished.emit(f"Processed {video_idx+1}/{len(video_files)}: {video_name}")
                
                if self.config.get('export_format', 'None') != 'None':
                    self.export_results(video_save_dir, self.detection_results, loaded_models[0].names)
                
                self.create_performance_report(video_save_dir)
                self.detection_results = {}
                
            self.detection_finished.emit(str(save_path))
        else:
            # 단일 소스 처리
            save_dir = Path(increment_path(Path(save_path), exist_ok=True)) 
            (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
            for model_name in self.model_performances:
                self.model_performances[model_name]['class_counts'] = {}
            
            self.process_single_source(
                source, loaded_models, model_names, model_filters,
                img_size, stride, conf_thres, iou_thres, device, half,
                save_dir, view_img, save_img, num_active_models
            )
            
            if self.config.get('export_format', 'None') != 'None':
                self.export_results(save_dir, self.detection_results, loaded_models[0].names)
                
            self.create_performance_report(save_dir)
            self.detection_finished.emit(str(save_dir))

    def process_single_source(self, source, loaded_models, model_names, model_filters,
                            img_size, stride, conf_thres, iou_thres, device, half,
                            save_dir, view_img, save_img, num_active_models):
        """개선된 단일 소스 처리 - 메모리 최적화 적용"""
        
        # 기본 설정
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        
        if webcam:
            dataset = LoadStreams(source, img_size=img_size, stride=stride)
        else:
            dataset = LoadImages(source, img_size=img_size, stride=stride)
        
        names = loaded_models[0].module.names if hasattr(loaded_models[0], 'module') else loaded_models[0].names
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
        
        # GPU 워밍업
        if device.type != 'cpu':
            print("🔥 GPU 워밍업 중...")
            for model in loaded_models:
                model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
        
        # 비디오 writer 관련 변수
        vid_path, vid_writer = None, None
        vid_writer_config = None
        start_time = time.time()
        processed_frames = 0
        
        # 샘플 캡처 초기화
        self.captured_samples = 0
        samples_dir = Path(save_dir) / 'samples'
        if self.enable_samples:
            samples_dir.mkdir(parents=True, exist_ok=True)
            print(f"📸 샘플 캡처 활성화: 최대 {self.max_samples}개")
        
        # 처리 루프
        total_frames = 0
        frame_count = 0
        frame_interval = 1
        first_iter = True
        
        for path, img, im0s, vid_cap in dataset:
            if not self.running:
                break
            
            # 첫 번째 반복에서 비디오 정보 확인
            if first_iter and self.target_fps > 0 and vid_cap and not webcam:
                original_fps = vid_cap.get(cv2.CAP_PROP_FPS)
                if original_fps > 0:
                    frame_interval = max(1, round(original_fps / self.target_fps))
                    print(f"📺 원본 FPS: {original_fps:.1f}, 목표 FPS: {self.target_fps}, "
                          f"처리 간격: 매 {frame_interval}번째 프레임")
                first_iter = False
                
            # 총 프레임 수 계산
            if total_frames == 0 and not webcam and vid_cap:
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"🎞️ 총 프레임 수: {total_frames}")
                
            frame_count += 1
            self.frame_counter += 1
            
            # FPS 제어
            if self.target_fps > 0 and frame_count % frame_interval != 0:
                continue
                
            processed_frames += 1
            
            # 주기적 메모리 정리
            if self.frame_counter % self.memory_check_interval == 0:
                if not self.check_memory_usage():
                    print("⏳ 메모리 정리 중...")
                    time.sleep(0.05)  # 짧은 대기
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 실제 FPS 계산
            if processed_frames % 30 == 0:  # 30프레임마다
                elapsed = time.time() - start_time
                if elapsed > 0:
                    actual_fps = processed_frames / elapsed
                    for model_name in self.model_performances:
                        self.model_performances[model_name]['actual_fps'] = actual_fps

            # 추론 처리
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 모든 모델 추론
            all_preds = []
            with torch.no_grad():
                for i, model in enumerate(loaded_models):
                    try:
                        t0 = time_synchronized()
                        pred = model(img)[0]
                        pred = non_max_suppression(pred, conf_thres, iou_thres)
                        t1 = time_synchronized()
                        
                        inference_time = t1 - t0
                        model_name = model_names[i]
                        self.model_performances[model_name]['inference_times'].append(inference_time)
                        
                        # 클래스별 빈도수 계산
                        for j, det in enumerate(pred):
                            if len(det):
                                for *_, conf, cls in det:
                                    cls_id = int(cls.item())
                                    if 0 <= cls_id < len(names):
                                        cls_name = names[cls_id]
                                        if cls_name not in self.model_performances[model_name]['class_counts']:
                                            self.model_performances[model_name]['class_counts'][cls_name] = 0
                                        self.model_performances[model_name]['class_counts'][cls_name] += 1
                        
                        # 필터 적용
                        if model_filters.get(i):
                            filtered_classes = model_filters[i]
                            for j, det in enumerate(pred):
                                if len(det):
                                    mask = torch.zeros(det.shape[0], dtype=torch.bool)
                                    for k, (*_, cls) in enumerate(det):
                                        if int(cls) in filtered_classes:
                                            mask[k] = True
                                    pred[j] = det[mask]
                        
                        total_detections = sum(len(d) for d in pred)
                        self.model_performances[model_name]['detections_count'] += total_detections
                        all_preds.append(pred)
                        
                        # FPS 계산 및 업데이트
                        if inference_time > 0:
                            current_fps = 1.0 / inference_time
                            # 지수 이동 평균으로 FPS 계산
                            if self.model_performances[model_name]['fps'] == 0:
                                self.model_performances[model_name]['fps'] = current_fps
                            else:
                                alpha = 0.1  # 스무딩 팩터
                                self.model_performances[model_name]['fps'] = (
                                    alpha * current_fps + 
                                    (1 - alpha) * self.model_performances[model_name]['fps']
                                )
                        
                    except torch.cuda.OutOfMemoryError:
                        print(f"🚨 GPU 메모리 부족, 모델 {model_name} 건너뜀")
                        torch.cuda.empty_cache()
                        continue
                    except Exception as e:
                        print(f"❌ 모델 {model_name} 추론 중 오류: {e}")
                        continue
            
            # 검출 결과 처리
            for i, batch_dets in enumerate(zip(*all_preds)):
                if webcam:
                    p, frame = path[i], dataset.count
                    im0s_copy = self.safe_image_copy(im0s[i])
                else:
                    p, frame = path, getattr(dataset, 'frame', 0)
                    im0s_copy = self.safe_image_copy(im0s)
                
                # 검출 결과 처리
                processed_images = self.process_detections_adaptive(
                    batch_dets, im0s_copy, model_names, names, colors, 
                    save_img, view_img, frame
                )
                
                if processed_images:
                    # 적응적 그리드 생성
                    grid_image = self.resolution_manager.create_adaptive_grid(
                        processed_images, max_width=1920, max_height=1080
                    )
                    
                    if grid_image is not None:
                        # 샘플 캡처 처리
                        if self.enable_samples and self.captured_samples < self.max_samples:
                            self.handle_sample_capture(batch_dets, model_names, frame_count, 
                                                     samples_dir, grid_image)
                        
                        # UI 업데이트
                        self.update_frame.emit(grid_image)
                        
                        # 비디오 저장 처리
                        if save_img and dataset.mode != 'image':
                            save_path = str(save_dir / Path(p).name)
                            
                            # 비디오 writer 설정 (해상도 변경시 재설정)
                            current_config = (grid_image.shape[1], grid_image.shape[0])
                            if vid_writer_config != current_config:
                                if vid_writer:
                                    vid_writer.release()
                                    print(f"📹 비디오 writer 재설정: {current_config}")
                                
                                fps = self.target_fps if self.target_fps > 0 else (
                                    vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25.0
                                )
                                
                                vid_writer = cv2.VideoWriter(
                                    save_path + '.mp4',
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps, current_config
                                )
                                vid_writer_config = current_config
                            
                            if vid_writer and vid_writer.isOpened():
                                vid_writer.write(grid_image)
                        
                        # 진행률 업데이트
                        if not webcam and total_frames > 0:
                            progress = int((frame / total_frames) * 100)
                            self.update_progress.emit(progress)
        
        # 정리
        if vid_writer:
            vid_writer.release()
            print("📹 비디오 저장 완료")
        
        # 최종 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ 처리 완료: {processed_frames}프레임 처리됨")

    def handle_sample_capture(self, batch_dets, model_names, frame_count, samples_dir, grid_image):
        """샘플 캡처 처리"""
        target_detected = False
        detecting_model_idx = None
        
        for j, (det, model_name) in enumerate(zip(batch_dets, model_names)):
            # 특정 모델에서만 감지하도록 설정된 경우
            if self.sample_model_id is not None and j != self.sample_model_id:
                continue
                
            # 특정 클래스 감지 확인
            if len(det) > 0:
                if self.target_class_id is not None:
                    for *_, conf, cls in det:
                        if int(cls.item()) == self.target_class_id:
                            target_detected = True
                            detecting_model_idx = j
                            break
                else:
                    # 모든 클래스 대상인 경우
                    target_detected = True
                    detecting_model_idx = j
            
            if target_detected:
                break
                
        # 타겟 조건을 만족하면 샘플 저장
        if target_detected:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            class_info = f"class{self.target_class_id}" if self.target_class_id is not None else "any_class"
            model_info = f"model{detecting_model_idx}" if detecting_model_idx is not None else "any_model"
            sample_file = samples_dir / f"sample_{timestamp}_{class_info}_{model_info}_frame{frame_count:04d}.jpg"
            
            try:
                cv2.imwrite(str(sample_file), grid_image)
                self.captured_samples += 1
                print(f"📸 샘플 캡처 {self.captured_samples}/{self.max_samples}: {sample_file.name}")
            except Exception as e:
                print(f"❌ 샘플 캡처 중 오류: {str(e)}")

    def export_results(self, save_dir, detections_per_model, names):
        """결과 내보내기"""
        export_format = self.config.get('export_format', 'None')
        if export_format == 'None':
            return
            
        print(f"📤 결과 내보내기: {export_format} 형식")
        
        if export_format == 'CSV':
            import csv
            with open(os.path.join(save_dir, 'detections.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['model', 'frame', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
                
                for model_name, frames in detections_per_model.items():
                    for frame_idx, detections in frames.items():
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            writer.writerow([
                                model_name, 
                                frame_idx, 
                                names[det['cls']], 
                                det['conf'], 
                                x1, y1, x2, y2
                            ])
        
        elif export_format == 'JSON':
            import json
            results = []
            
            for model_name, frames in detections_per_model.items():
                for frame_idx, detections in frames.items():
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        results.append({
                            'model': model_name,
                            'frame': frame_idx,
                            'class': names[det['cls']],
                            'confidence': det['conf'],
                            'bbox': [x1, y1, x2, y2]
                        })
            
            with open(os.path.join(save_dir, 'detections.json'), 'w') as jsonfile:
                json.dump(results, jsonfile, indent=4)

    def create_performance_report(self, save_dir):
        """성능 보고서 생성"""
        report_path = os.path.join(save_dir, 'performance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== YOLOv7 메모리 최적화 성능 보고서 ===\n\n")
            f.write(f"생성 날짜: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 시스템 정보
            memory_info = get_memory_usage_info()
            f.write("시스템 정보:\n")
            f.write(f"  총 메모리: {memory_info['total_gb']:.1f}GB\n")
            f.write(f"  사용 가능 메모리: {memory_info['available_gb']:.1f}GB\n")
            f.write(f"  메모리 사용률: {memory_info['percent']:.1f}%\n")
            
            if 'gpu' in memory_info:
                f.write(f"  GPU 메모리: {memory_info['gpu']['total_gb']:.1f}GB\n")
                f.write(f"  GPU 사용량: {memory_info['gpu']['allocated_gb']:.1f}GB\n")
            f.write("\n")
            
            # 처리 설정
            if self.enable_samples:
                f.write("샘플 캡처 설정:\n")
                f.write(f"  최대 샘플 수: {self.max_samples}\n")
                f.write(f"  타겟 클래스: {self.target_class}\n")
                f.write(f"  소스 모델: {self.sample_model}\n")
                f.write(f"  캡처된 샘플: {self.captured_samples}\n\n")
            
            if self.target_fps > 0:
                f.write(f"목표 FPS: {self.target_fps}\n")
            else:
                f.write("목표 FPS: 제한 없음\n")
            f.write("\n")
            
            # 각 모델별 성능 데이터
            for model_name, perf in self.model_performances.items():
                f.write(f"모델: {model_name}\n")
                f.write(f"  추론 FPS: {perf['fps']:.2f}\n")
                f.write(f"  실제 FPS: {perf.get('actual_fps', 0):.2f}\n")
                
                if perf['inference_times']:
                    avg_time = sum(perf['inference_times']) / len(perf['inference_times'])
                    f.write(f"  평균 추론 시간: {avg_time*1000:.2f} ms\n")
                    avg_detections = perf['detections_count'] / len(perf['inference_times'])
                    f.write(f"  프레임당 평균 감지 수: {avg_detections:.2f}\n")
                
                f.write(f"  총 감지된 객체: {perf['detections_count']}\n")
                
                # 클래스별 빈도수
                if perf['class_counts']:
                    f.write("\n  클래스별 감지 빈도:\n")
                    for cls_name, count in sorted(perf['class_counts'].items(), 
                                                key=lambda x: x[1], reverse=True):
                        f.write(f"    {cls_name}: {count}\n")
                
                f.write("\n")
            
            # 하드웨어 정보
            f.write("하드웨어 정보:\n")
            if torch.cuda.is_available():
                f.write(f"  CUDA 장치: {torch.cuda.get_device_name(0)}\n")
                f.write(f"  CUDA 버전: {torch.version.cuda}\n")
            else:
                f.write("  CPU에서 실행\n")
        
        print(f"📊 성능 보고서 생성: {report_path}")

    def stop(self):
        """스레드 중지"""
        print("⏹️ 처리 중지 요청됨")
        self.running = False

class ModelSelector(QWidget):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.parent = parent  # 부모 윈도우에 대한 참조 추가
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(f"Model {index+1}:")
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_model)

        self.view_classes_btn = QPushButton("View Classes")
        self.view_classes_btn.clicked.connect(self.view_classes)
        self.view_classes_btn.setEnabled(False)
        
        # 추가: 필터 버튼
        self.filter_classes_btn = QPushButton("Filter Classes")
        self.filter_classes_btn.clicked.connect(self.filter_classes)
        self.filter_classes_btn.setEnabled(False)

        self.enabled_checkbox = QCheckBox("Enable")
        self.enabled_checkbox.setChecked(index == 0)  # First model is enabled by default
        self.enabled_checkbox.stateChanged.connect(self.toggle_enabled)
        
        self.filtered_classes = []

        layout.addWidget(label)
        layout.addWidget(self.path_edit)
        layout.addWidget(browse_btn)
        layout.addWidget(self.view_classes_btn)
        layout.addWidget(self.filter_classes_btn)  # 추가된 버튼
        layout.addWidget(self.enabled_checkbox)
        
        self.setLayout(layout)
        self.toggle_enabled(self.enabled_checkbox.isChecked())
        
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Model {self.index+1}", "", "YOLOv7 Model (*.pt)"
        )
        if file_path:
            self.path_edit.setText(file_path)
            self.enabled_checkbox.setChecked(True)
            self.view_classes_btn.setEnabled(True)
            self.filter_classes_btn.setEnabled(True)  # 필터 버튼 활성화

    
    def toggle_enabled(self, state):
        enabled = bool(state)
        self.path_edit.setEnabled(enabled)
        has_model_path = bool(self.path_edit.text())
        self.view_classes_btn.setEnabled(enabled and has_model_path)
        self.filter_classes_btn.setEnabled(enabled and has_model_path)  # 필터 버튼 상태 업데이트

    def filter_classes(self):
        model_path = self.path_edit.text()
        if model_path and os.path.exists(model_path):
            dialog = ClassFilterDialog(model_path, self.filtered_classes, self.parent)
            if dialog.exec_():
                self.filtered_classes = dialog.get_filtered_classes()
                # 필터 상태 표시
                if self.filtered_classes:
                    self.path_edit.setToolTip(f"Filtered: {len(self.filtered_classes)} classes selected")
                    self.filter_classes_btn.setStyleSheet("background-color: #AAFFAA;")
                else:
                    self.path_edit.setToolTip("")
                    self.filter_classes_btn.setStyleSheet("")

    def get_model_path(self):
        if self.enabled_checkbox.isChecked() and self.path_edit.text():
            return self.path_edit.text()
        return ""
        
    def get_model_info(self):
        if self.enabled_checkbox.isChecked() and self.path_edit.text():
            return {
                'path': self.path_edit.text(),
                'filtered_classes': self.filtered_classes
            }
        return None
        
    def view_classes(self):
        model_path = self.path_edit.text()
        if model_path and os.path.exists(model_path):
            dialog = ModelClassInfo(model_path, self.parent)
            dialog.exec_()

def get_memory_usage_info():
    """상세한 메모리 사용량 정보"""
    memory = psutil.virtual_memory()
    
    info = {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent,
        'status': 'critical' if memory.percent > 90 else 'warning' if memory.percent > 80 else 'normal'
    }
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        
        info['gpu'] = {
            'total_gb': gpu_memory,
            'allocated_gb': allocated,
            'cached_gb': cached,
            'free_gb': gpu_memory - cached
        }
    
    return info
def optimize_torch_settings():
    """PyTorch 메모리 최적화 설정"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("🚀 PyTorch 최적화 설정 완료")

class YOLOv7UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv7 Multi-Model Object Detection")
        self.setMinimumSize(1200, 800)
        
        self.detection_thread = None  # 처음에는 None으로 초기화
        self.model_selectors = []
        self.last_results = {}  # 모델별 마지막 결과 저장
        self.video_files = []  # 처리할 비디오 파일 목록 추가
        self.initUI()
        
    def initUI(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 탭 위젯 추가
        self.tabs = QTabWidget()
        main_tab_widget = QWidget()
        self.main_tab_layout = QVBoxLayout(main_tab_widget)
        
        # ==========================================
        # Detection 탭 내용 추가
        # ==========================================
        
        # Create scrollable area for settings
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        
        # Model selection group
        model_group = QGroupBox("Model Selection (up to 6 models)")
        model_layout = QVBoxLayout()
        
        # Create model selectors
        for i in range(6):  # Up to 6 models
            model_selector = ModelSelector(i, self)
            self.model_selectors.append(model_selector)
            model_layout.addWidget(model_selector)
        
        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)
        
        # Source selection group
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout()
        
        # 소스 타입 선택 추가 (파일/폴더)
        source_type_layout = QHBoxLayout()
        source_type_label = QLabel("Source Type:")
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["File", "Directory"])
        self.source_type_combo.currentIndexChanged.connect(self.update_source_ui)
        source_type_layout.addWidget(source_type_label)
        source_type_layout.addWidget(self.source_type_combo)
        source_layout.addLayout(source_type_layout)
        
        # Source path
        source_path_layout = QHBoxLayout()
        self.source_label = QLabel("Source File:")
        self.source_path = QLineEdit()
        self.source_path.setReadOnly(True)
        source_browse = QPushButton("Browse")
        source_browse.clicked.connect(self.browse_source)
        source_path_layout.addWidget(self.source_label)
        source_path_layout.addWidget(self.source_path)
        source_path_layout.addWidget(source_browse)
        source_layout.addLayout(source_path_layout)
        
        # 파일 목록 위젯 (디렉토리 모드에서 표시)
        self.file_list_group = QGroupBox("Files to Process")
        file_list_layout = QVBoxLayout()
        self.file_list = QListView()
        self.file_list_model = QStringListModel()
        self.file_list.setModel(self.file_list_model)
        file_list_layout.addWidget(self.file_list)
        self.file_list_group.setLayout(file_list_layout)
        self.file_list_group.setVisible(False)  # 처음에는 숨김
        source_layout.addWidget(self.file_list_group)
        
        # Webcam option
        webcam_layout = QHBoxLayout()
        webcam_label = QLabel("Use Webcam:")
        self.webcam_combo = QComboBox()
        self.webcam_combo.addItem("No")
        self.webcam_combo.addItem("Webcam 0")
        self.webcam_combo.addItem("Webcam 1")
        self.webcam_combo.currentIndexChanged.connect(self.update_source_from_webcam)
        webcam_layout.addWidget(webcam_label)
        webcam_layout.addWidget(self.webcam_combo)
        source_layout.addLayout(webcam_layout)
        
        source_group.setLayout(source_layout)
        scroll_layout.addWidget(source_group)
        
        # Settings group
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QHBoxLayout()
        
        # Image size
        img_size_layout = QVBoxLayout()
        img_size_label = QLabel("Image Size:")
        self.img_size_combo = QComboBox()
        for size in [320, 416, 512, 640, 736, 1280]:
            self.img_size_combo.addItem(str(size))
        self.img_size_combo.setCurrentText("640")  # Default
        img_size_layout.addWidget(img_size_label)
        img_size_layout.addWidget(self.img_size_combo)
        settings_layout.addLayout(img_size_layout)

        target_fps_layout = QVBoxLayout()
        target_fps_label = QLabel("Target FPS:")
        self.target_fps_combo = QComboBox()
        self.target_fps_combo.addItems(["No Limit", "1", "2", "5", "10", "15", "20", "25", "30"])
        self.target_fps_combo.setCurrentText("No Limit")  # 기본값
        target_fps_layout.addWidget(target_fps_label)
        target_fps_layout.addWidget(self.target_fps_combo)
        settings_layout.addLayout(target_fps_layout)

        # Confidence threshold
        conf_layout = QVBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(25)  # Default 0.25
        self.conf_value = QLabel("0.25")
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)
        settings_layout.addLayout(conf_layout)
        
        # IOU threshold
        iou_layout = QVBoxLayout()
        iou_label = QLabel("IOU Threshold:")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(99)
        self.iou_slider.setValue(45)  # Default 0.45
        self.iou_value = QLabel("0.45")
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_value)
        settings_layout.addLayout(iou_layout)
        
        # Device selection
        device_layout = QVBoxLayout()
        device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        # Check for CUDA availability
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device_combo.addItem(f"CUDA:{i}")
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        settings_layout.addLayout(device_layout)
        
        settings_group.setLayout(settings_layout)
        scroll_layout.addWidget(settings_group)
        
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()

        # Save directory
        save_dir_layout = QHBoxLayout()
        save_dir_label = QLabel("Save Directory:")
        self.save_dir_path = QLineEdit()
        self.save_dir_path.setReadOnly(True)
        self.save_dir_path.setText(os.path.join(os.getcwd(), "runs/detect"))
        save_dir_browse = QPushButton("Browse")
        save_dir_browse.clicked.connect(lambda: self.browse_directory(self.save_dir_path))
        save_dir_layout.addWidget(save_dir_label)
        save_dir_layout.addWidget(self.save_dir_path)
        save_dir_layout.addWidget(save_dir_browse)
        output_layout.addLayout(save_dir_layout)

        # Checkboxes for options
        options_layout = QHBoxLayout()
        self.save_results = QCheckBox("Save Detection Results")
        self.save_results.setChecked(True)
        options_layout.addWidget(self.save_results)
        output_layout.addLayout(options_layout)

        # 내보내기 옵션 추가
        export_layout = QHBoxLayout()
        export_label = QLabel("Export Results:")
        self.export_combo = QComboBox()
        self.export_combo.addItem("None")
        self.export_combo.addItem("CSV")
        self.export_combo.addItem("JSON")
        export_layout.addWidget(export_label)
        export_layout.addWidget(self.export_combo)
        output_layout.addLayout(export_layout)

        # 샘플 캡처 설정 그룹 추가
        sample_group = QGroupBox("Sample Capture Settings")
        sample_group_layout = QVBoxLayout()

        # 샘플 활성화 및 개수 설정
        sample_config_layout = QHBoxLayout()
        self.enable_samples = QCheckBox("Enable Samples")
        sample_config_layout.addWidget(self.enable_samples)

        sample_config_layout.addWidget(QLabel("Max Samples:"))
        self.max_samples = QLineEdit("10")  # 기본값 10개
        self.max_samples.setFixedWidth(50)
        sample_config_layout.addWidget(self.max_samples)
        sample_group_layout.addLayout(sample_config_layout)

        # 타겟 클래스 설정
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Target Class:"))
        self.target_class = QComboBox()
        self.target_class.setEditable(True)
        self.target_class.addItem("Any Class")  # 기본값 - 모든 클래스
        class_layout.addWidget(self.target_class)
        sample_group_layout.addLayout(class_layout)

        # 모델 선택 설정
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Detect from:"))
        self.sample_model_combo = QComboBox()
        self.sample_model_combo.addItem("Any Model")  # 기본값 - 어떤 모델이든
        model_layout.addWidget(self.sample_model_combo)
        sample_group_layout.addLayout(model_layout)

        sample_group.setLayout(sample_group_layout)
        output_layout.addWidget(sample_group)

        # Output 그룹 완성
        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)
        
        # save_dir_path에 대한 이벤트 연결
        self.save_dir_path.textChanged.connect(self.check_summary_button_state)
        
        # 스크롤 영역을 메인 탭 레이아웃에 추가
        self.main_tab_layout.addWidget(scroll_area)
        
        # Display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Detection results will appear here")
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.image_label.setMinimumHeight(400)
        self.main_tab_layout.addWidget(self.image_label)
        
        # 성능 표시를 위한 그룹 상자 추가
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QVBoxLayout()
        
        # 모델별 성능 표시 라벨
        self.perf_labels = {}
        for i in range(6):  # 최대 6개 모델
            label = QLabel(f"Model {i+1}: No data")
            label.setVisible(False)
            perf_layout.addWidget(label)
            self.perf_labels[i] = label
        
        perf_group.setLayout(perf_layout)
        self.main_tab_layout.addWidget(perf_group)
        
        # 로그 영역 추가 (비디오 처리 진행상황 표시)
        self.log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_text = QLabel("Waiting to start processing...")
        log_layout.addWidget(self.log_text)
        self.log_group.setLayout(log_layout)
        self.main_tab_layout.addWidget(self.log_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_tab_layout.addWidget(self.progress_bar)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.clicked.connect(self.take_snapshot)
        self.snapshot_button.setEnabled(False)
        self.compare_button = QPushButton("Compare Results")
        self.compare_button.clicked.connect(self.show_comparison)
        self.compare_button.setEnabled(False)
        
        # 요약 버튼 추가
        self.summary_button = QPushButton("Show Results Summary")
        self.summary_button.clicked.connect(self.show_summary)
        self.summary_button.setEnabled(False)
        
        # 모든 버튼을 한 번에 추가
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.snapshot_button)
        buttons_layout.addWidget(self.compare_button)
        buttons_layout.addWidget(self.summary_button)
        self.main_tab_layout.addLayout(buttons_layout)
        
        # ==========================================
        # 분석 탭 설정
        # ==========================================
        
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        
        # 분석 옵션 컨트롤
        analysis_controls = QHBoxLayout()
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["Class Distribution", "Detection Confidence", "Detection Size", "Performance Metrics"])
        self.analysis_type_combo.currentIndexChanged.connect(self.update_analysis_chart)
        analysis_controls.addWidget(QLabel("Analysis Type:"))
        analysis_controls.addWidget(self.analysis_type_combo)
        analysis_layout.addLayout(analysis_controls)
        
        # matplotlib 그래프
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        analysis_layout.addWidget(self.canvas)
        
        # 탭 위젯에 탭 추가
        self.tabs.addTab(main_tab_widget, "Detection")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        
        # main_layout에 탭 위젯 추가
        main_layout.addWidget(self.tabs)
        
        # 분석 데이터 저장 변수
        self.analysis_data = {
            'class_counts': {},  # 클래스별 감지 횟수
            'confidences': {},   # 클래스별 신뢰도 목록
            'box_sizes': {},     # 클래스별 박스 크기 목록
            'inference_times': []  # 추론 시간 목록
        }
        
        # 초기 상태 확인
        self.check_summary_button_state()

        # 비디오 처리 완료 신호 연결
        # 이 부분은 detection_thread가 생성될 때 연결됩니다
    
    def update_source_ui(self, index):
        """소스 유형(파일/폴더)에 따라 UI 업데이트"""
        is_directory = index == 1  # 1 = Directory
        
        # 소스 레이블 업데이트
        self.source_label.setText("Source Directory:" if is_directory else "Source File:")
        
        # 파일 목록 위젯 표시/숨김
        self.file_list_group.setVisible(is_directory)
        
        # 소스 경로 초기화
        self.source_path.setText("")
        self.file_list_model.setStringList([])
        self.video_files = []
        
        # 웹캠 옵션 비활성화 (디렉토리 모드)
        self.webcam_combo.setEnabled(not is_directory)
        if is_directory:
            self.webcam_combo.setCurrentIndex(0)  # 웹캠 사용 안함으로 설정
    
    def browse_source(self):
        if self.source_type_combo.currentText() == "Directory":
            # 디렉토리 선택
            directory = QFileDialog.getExistingDirectory(self, "Select Video Directory")
            if directory:
                self.source_path.setText(directory)
                self.webcam_combo.setCurrentIndex(0)  # Set to "No" when directory is selected
                
                # 디렉토리 내 비디오 파일 검색
                self.scan_video_files(directory)
        else:
            # 파일 선택 (기존 로직)
            source_path, _ = QFileDialog.getOpenFileName(
                self, "Select Source", "", "Images/Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov)"
            )
            if source_path:
                self.source_path.setText(source_path)
                self.webcam_combo.setCurrentIndex(0)  # Set to "No" when file is selected
    
    def scan_video_files(self, directory):
        """지정된 디렉토리에서 비디오 파일을 검색합니다"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        # 디렉토리 내 모든 파일 스캔
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(root, file)
                    video_files.append(video_path)
        
        # 파일 목록 업데이트
        self.video_files = video_files
        self.file_list_model.setStringList([os.path.basename(f) for f in video_files])
        
        # 로그 업데이트
        self.log_text.setText(f"Found {len(video_files)} video files in the directory.")
    
    def browse_directory(self, line_edit):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)
    
    def update_source_from_webcam(self, index):
        if index > 0:  # Webcam selected
            self.source_path.setText(str(index - 1))  # 0 for Webcam 0, 1 for Webcam 1
        else:
            self.source_path.setText("")
    
    def update_conf_label(self):
        value = self.conf_slider.value() / 100.0
        self.conf_value.setText(f"{value:.2f}")
    
    def update_iou_label(self):
        value = self.iou_slider.value() / 100.0
        self.iou_value.setText(f"{value:.2f}")
    
    def start_detection(self):
        # 폴더 모드인지 확인
        is_directory = self.source_type_combo.currentText() == "Directory"
        
        # Get models with filter info
        models = []
        for selector in self.model_selectors:
            if hasattr(selector, 'get_model_info'):
                model_info = selector.get_model_info()
                if model_info:
                    models.append(model_info)
            else:
                # 이전 버전과의 호환성을 위해
                model_path = selector.get_model_path()
                if model_path:
                    models.append(model_path)
        
        active_models = [m for m in models if m]
        
        # Validate inputs
        if not active_models:
            QMessageBox.warning(self, "Missing Input", "Please select at least one model.")
            return
        
        if not self.source_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select a source (image, video, or webcam).")
            return
            
        if is_directory and not self.video_files:
            QMessageBox.warning(self, "No Videos Found", "No video files found in the selected directory.")
            return
        
        # Create configuration for detection thread
        config = {
            'models': models,
            'source': self.source_path.text(),
            'img_size': int(self.img_size_combo.currentText()),
            'conf_thres': float(self.conf_value.text()),
            'iou_thres': float(self.iou_value.text()),
            'device': '' if self.device_combo.currentText() == "CPU" else self.device_combo.currentText().split(':')[1],
            'save_path': self.save_dir_path.text(),
            'view_img': True,
            'save_img': self.save_results.isChecked(),
            'export_format': self.export_combo.currentText(),
            # Target FPS 설정 추가
            'target_fps': 0 if self.target_fps_combo.currentText() == "No Limit" else float(self.target_fps_combo.currentText()),
        }
        
        # 폴더 모드인 경우 비디오 파일 목록 추가
        if is_directory:
            config['is_directory'] = True
            config['video_files'] = self.video_files
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.log_text.setText("Initializing detection...")
        
        # 이전 결과 초기화
        self.last_results = {}
        
        self.detection_thread = OptimizedDetectionThread(config, self)
        self.detection_thread.update_frame.connect(self.update_display)
        self.detection_thread.update_progress.connect(self.update_progress)
        self.detection_thread.update_performance.connect(self.update_performance_display)
        self.detection_thread.update_analysis_data.connect(self.update_analysis_data)
        self.detection_thread.detection_finished.connect(self.detection_completed)
        
        # 비디오 처리 완료 신호 연결
        self.detection_thread.video_finished.connect(self.video_processing_completed)

        config['enable_samples'] = self.enable_samples.isChecked()
        try:
            config['max_samples'] = int(self.max_samples.text())
        except ValueError:
            config['max_samples'] = 10  # 기본값
        
        config['target_class'] = self.target_class.currentText()
        config['sample_model'] = self.sample_model_combo.currentText()
        
        self.detection_thread.start()
    
    def video_processing_completed(self, message):
        """개별 비디오 처리 완료 시 호출되는 함수"""
        self.log_text.setText(message)
    
    def stop_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait()
            self.detection_completed("Detection stopped by user")
    
    def update_display(self, frame):
        # 현재 프레임 저장 (BGR 형식)
        self.current_frame = frame.copy()
        
        # 스냅샷 버튼 활성화 - UI 업데이트는 메인 스레드에서만 수행해야 함
        # Qt에서는 모든 UI 작업은 메인 스레드에서 이루어져야 함
        # self.snapshot_button.setEnabled(True) - 여기서 직접 호출하지 않고 메소드를 통해 호출
        QTimer.singleShot(0, lambda: self.snapshot_button.setEnabled(True))
        
        # Convert from OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        # Create QImage and QPixmap
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        
        # Resize to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the pixmap - UI 업데이트이므로 메인 스레드에서 처리
        QTimer.singleShot(0, lambda: self.image_label.setPixmap(pixmap))
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def detection_completed(self, result_path):
        # UI 상태 업데이트는 메인 스레드에서 처리
        QTimer.singleShot(0, lambda: self._detection_completed_internal(result_path))
    def show_summary(self):
        """결과 요약 대화상자 표시"""
        # 저장 경로 확인
        save_dir = self.save_dir_path.text()
        if not save_dir or not os.path.exists(save_dir):
            QMessageBox.warning(self, "No Results", "No results directory found.")
            return
            
        # 결과 폴더 내에 performance_report.txt 파일이 있는지 확인
        has_reports = False
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if file == 'performance_report.txt':
                    has_reports = True
                    break
            if has_reports:
                break
                
        if not has_reports:
            QMessageBox.warning(self, "No Reports", "No performance reports found in the results directory.")
            return
            
        # 요약 대화상자 표시
        summary_dialog = ResultsSummaryDialog(save_dir, self)
        summary_dialog.exec_()
    def check_summary_button_state(self):
        """저장 경로에 결과 파일이 있는지 확인하고 요약 버튼 상태 업데이트"""
        save_dir = self.save_dir_path.text()
        
        # 기본 비활성화
        self.summary_button.setEnabled(False)
        
        if save_dir and os.path.exists(save_dir):
            # 결과 폴더 내에 performance_report.txt 파일이 있는지 확인
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    if file == 'performance_report.txt':
                        self.summary_button.setEnabled(True)
                        return
    
    def _detection_completed_internal(self, result_path):
        """UI 업데이트 로직을 처리하는 내부 메소드 (메인 스레드에서 호출)"""
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.snapshot_button.setEnabled(False)
        
        # 비교 버튼 활성화 (결과가 있을 경우)
        if self.last_results:
            self.compare_button.setEnabled(True)

        self.summary_button.setEnabled(True)

        # 로그 업데이트
        self.log_text.setText(f"All processing completed. Results saved to: {result_path}")
        
        # Show result message
        QMessageBox.information(self, "Detection Complete", f"Detection completed.\nResults saved to: {result_path}")
        
        # Hide progress bar after 3 seconds
        QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))
        
        # 디렉토리 모드이고 처리된 비디오가 여러 개인 경우, 요약 대화상자 표시
        if self.source_type_combo.currentText() == "Directory" and len(self.video_files) > 1:
            summary_dialog = ResultsSummaryDialog(result_path, self)
            summary_dialog.exec_()
        
    def update_performance_display(self, perf_data):
        for i, model_selector in enumerate(self.model_selectors):
            model_path = model_selector.get_model_path() if hasattr(model_selector, 'get_model_path') else None
            if not model_path:
                self.perf_labels[i].setVisible(False)
                continue
                
            model_name = os.path.basename(model_path)
            if model_name in perf_data:
                perf = perf_data[model_name]
                self.perf_labels[i].setText(
                    f"Model {i+1}: {model_name} - FPS: {perf['fps']:.2f}, "
                    f"Inference: {perf['avg_inference_time']*1000:.2f} ms, "
                    f"Detections: {perf['detections_count']}"
                )
                self.perf_labels[i].setVisible(True)
    
    def update_analysis_chart(self):
        # 선택된 분석 유형
        analysis_type = self.analysis_type_combo.currentText()
        
        # 차트 초기화
        self.axes.clear()
        
        if analysis_type == "Class Distribution":
            # 클래스별 감지 수 분석
            if self.analysis_data['class_counts']:
                classes = list(self.analysis_data['class_counts'].keys())
                counts = [self.analysis_data['class_counts'][cls] for cls in classes]
                
                # 막대 그래프 생성
                self.axes.bar(classes, counts)
                self.axes.set_xlabel('Class')
                self.axes.set_ylabel('Detection Count')
                self.axes.set_title('Class Distribution')
                
                # x축 라벨 회전 (클래스 이름이 길 경우)
                plt.setp(self.axes.get_xticklabels(), rotation=45, ha='right')
        
        elif analysis_type == "Detection Confidence":
            # 클래스별 신뢰도 분포 (박스 플롯)
            if self.analysis_data['confidences']:
                data = []
                labels = []
                
                for cls, conf_list in self.analysis_data['confidences'].items():
                    if conf_list:
                        data.append(conf_list)
                        labels.append(cls)
                
                if data:
                    self.axes.boxplot(data)
                    self.axes.set_xticklabels(labels)
                    self.axes.set_xlabel('Class')
                    self.axes.set_ylabel('Confidence')
                    self.axes.set_title('Detection Confidence Distribution')
        
        elif analysis_type == "Detection Size":
            # 클래스별 박스 크기 분포
            if self.analysis_data['box_sizes']:
                data = []
                labels = []
                
                for cls, size_list in self.analysis_data['box_sizes'].items():
                    if size_list:
                        data.append(size_list)
                        labels.append(cls)
                
                if data:
                    self.axes.boxplot(data)
                    self.axes.set_xticklabels(labels)
                    self.axes.set_xlabel('Class')
                    self.axes.set_ylabel('Box Size (pixels²)')
                    self.axes.set_title('Detection Size Distribution')
        
        elif analysis_type == "Performance Metrics":
            # 시간에 따른 추론 시간 변화
            if self.analysis_data['inference_times']:
                times = self.analysis_data['inference_times']
                x = range(len(times))
                
                self.axes.plot(x, times)
                self.axes.set_xlabel('Frame')
                self.axes.set_ylabel('Inference Time (ms)')
                self.axes.set_title('Inference Time per Frame')
        
        # 차트 업데이트
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_analysis_data(self, detections, class_names, inference_time):
        # 추론 시간 기록
        self.analysis_data['inference_times'].append(inference_time * 1000)  # ms 단위로 변환
        
        # 모든 감지 객체 처리
        for *xyxy, conf, cls in detections:
            cls_idx = int(cls)
            cls_name = class_names[cls_idx]
            
            # 클래스별 카운트 업데이트
            if cls_name not in self.analysis_data['class_counts']:
                self.analysis_data['class_counts'][cls_name] = 0
                self.analysis_data['confidences'][cls_name] = []
                self.analysis_data['box_sizes'][cls_name] = []
            
            self.analysis_data['class_counts'][cls_name] += 1
            
            # 신뢰도 기록
            self.analysis_data['confidences'][cls_name].append(float(conf))
            
            # 박스 크기 계산 및 기록
            x1, y1, x2, y2 = [int(c) for c in xyxy]
            box_area = (x2 - x1) * (y2 - y1)
            self.analysis_data['box_sizes'][cls_name].append(box_area)
        
        # 차트 업데이트 (30프레임마다 - 성능 최적화)
        if len(self.analysis_data['inference_times']) % 30 == 0:
            self.update_analysis_chart()
    
    def store_detection_result(self, frame, model_name, result_image):
        """감지 결과 이미지를 저장합니다"""
        if model_name not in self.last_results:
            self.last_results[model_name] = []
        
        # 필요한 경우 목록 확장
        while len(self.last_results[model_name]) <= frame:
            self.last_results[model_name].append(None)
        
        # 결과 저장
        self.last_results[model_name][frame] = result_image
    
    def show_comparison(self):
        """모델 비교 대화상자를 표시합니다"""
        if not self.last_results:
            QMessageBox.warning(self, "No Results", "Run detection first to get results for comparison.")
            return
        
        dialog = ComparisonViewerDialog(self.last_results, self)
        dialog.exec_()
    
    def take_snapshot(self):
        """현재 화면의 스냅샷을 저장합니다"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "No Image", "No detection image available to capture.")
            return
        
        # 저장 경로 선택
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", "", "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            try:
                # OpenCV 이미지로 저장
                cv2.imwrite(file_path, self.current_frame)
                QMessageBox.information(self, "Snapshot Saved", f"Snapshot saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save snapshot: {str(e)}")

    def update_class_list(self, model_path):
            """모델에서 클래스 목록을 추출하여 타겟 클래스 콤보박스 업데이트"""
            try:
                model = torch.load(model_path, map_location=torch.device('cpu'))
                class_names = []
                
                # 모델 형식에 따라 클래스 이름 추출
                if isinstance(model, dict):
                    if 'model' in model:
                        if hasattr(model['model'], 'names'):
                            class_names = model['model'].names
                        elif 'names' in model:
                            class_names = model['names']
                        else:
                            class_names = model['model'].module.names if hasattr(model['model'], 'module') else []
                    else:
                        for key in model:
                            if 'names' in str(key).lower():
                                class_names = model[key]
                                break
                else:
                    class_names = model.names if hasattr(model, 'names') else []
                
                if class_names:
                    # 현재 콤보박스 아이템 저장
                    current_class_text = self.target_class.currentText()
                    
                    # 클래스 콤보박스 업데이트
                    self.target_class.clear()
                    self.target_class.addItem("Any Class")  # 기본값으로 추가
                    for i, name in enumerate(class_names):
                        self.target_class.addItem(f"{i}: {name}")
                    
                    # 이전 선택 복원 시도
                    if current_class_text:
                        index = self.target_class.findText(current_class_text)
                        if index >= 0:
                            self.target_class.setCurrentIndex(index)
                
                # 모델 목록 업데이트
                self.update_sample_model_list()
                
            except Exception as e:
                print(f"클래스 목록 업데이트 중 오류: {str(e)}")


    def update_sample_model_list(self):
        # 현재 선택된 항목 저장
        current_text = self.sample_model_combo.currentText()
        
        # 모델 콤보박스 초기화
        self.sample_model_combo.clear()
        self.sample_model_combo.addItem("Any Model")  # 기본값 추가
        
        # 활성화된 모델 추가
        for i, selector in enumerate(self.model_selectors):
            if hasattr(selector, 'get_model_path') and selector.get_model_path() and selector.enabled_checkbox.isChecked():
                model_name = os.path.basename(selector.get_model_path())
                self.sample_model_combo.addItem(f"Model {i+1}: {model_name}")
        
        # 이전 선택 복원 시도
        if current_text:
            index = self.sample_model_combo.findText(current_text)
            if index >= 0:
                self.sample_model_combo.setCurrentIndex(index)
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Model {self.index+1}", "", "YOLOv7 Model (*.pt)"
        )
        if file_path:
            self.path_edit.setText(file_path)
            self.enabled_checkbox.setChecked(True)
            self.view_classes_btn.setEnabled(True)
            self.filter_classes_btn.setEnabled(True)
            
            # 클래스 목록 업데이트
            if hasattr(self.parent, 'update_class_list'):
                self.parent.update_class_list(file_path)

    def toggle_enabled(self, state):
        enabled = bool(state)
        self.path_edit.setEnabled(enabled)
        has_model_path = bool(self.path_edit.text())
        self.view_classes_btn.setEnabled(enabled and has_model_path)
        self.filter_classes_btn.setEnabled(enabled and has_model_path)
        
        # 활성화 상태 변경 시 모델 목록 업데이트
        if hasattr(self.parent, 'update_sample_model_list'):
            self.parent.update_sample_model_list()
def main():
    app = QApplication(sys.argv)
    window = YOLOv7UI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # PyTorch 최적화 설정 적용
    optimize_torch_settings()
    
    app = QApplication(sys.argv)
    window = YOLOv7UI()
    window.show()
    sys.exit(app.exec_())