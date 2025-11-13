# update_ui.py - 완전한 GUI로 업데이트

from pathlib import Path

def update_to_full_ui():
    """완전한 UI로 업데이트"""
    
    print("🔄 완전한 GUI로 업데이트 중...")
    
    # 완전한 main_window.py 코드
    full_ui_code = '''"""
YOLOv7 Training GUI - Complete Main Window
완전한 메인 윈도우 구현
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class MainWindow:
    """완전한 메인 GUI 윈도우 클래스"""
    
    def __init__(self, root, trainer, config_manager, model_manager):
        self.root = root
        self.trainer = trainer
        self.config_manager = config_manager
        self.model_manager = model_manager
        
        # 훈련 상태
        self.is_training = False
        self.current_metrics = {}
        
        # UI 변수들
        self.setup_variables()
        
        # UI 생성
        self.create_ui()
        
        # 콜백 등록
        self.setup_callbacks()
        
    def setup_variables(self):
        """UI 변수들 초기화"""
        # 데이터셋 설정
        self.dataset_path_var = tk.StringVar()
        self.model_config_var = tk.StringVar(value="cfg/training/yolov7.yaml")
        self.weights_path_var = tk.StringVar()
        self.image_size_var = tk.StringVar(value="640")
        
        # 훈련 파라미터
        self.epochs_var = tk.IntVar(value=300)
        self.batch_size_var = tk.IntVar(value=16)
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        self.workers_var = tk.IntVar(value=8)
        self.device_var = tk.StringVar(value="0")
        
        # 훈련 옵션
        self.cache_images_var = tk.BooleanVar()
        self.multi_scale_var = tk.BooleanVar()
        self.image_weights_var = tk.BooleanVar()
        self.rect_var = tk.BooleanVar()
        self.adam_var = tk.BooleanVar()
        self.sync_bn_var = tk.BooleanVar()
        
        # 출력 설정
        self.project_name_var = tk.StringVar(value="runs/train")
        self.experiment_name_var = tk.StringVar(value="exp")
        
        # 진행 상태
        self.progress_var = tk.DoubleVar()
        self.status_text_var = tk.StringVar(value="훈련 대기 중...")
    
    def create_ui(self):
        """메인 UI 생성"""
        self.root.title("🚀 YOLOv7 Training GUI - Professional Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # 헤더 생성
        self.create_header()
        
        # 노트북 탭 생성
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 탭들 생성
        self.create_settings_tab()
        self.create_progress_tab()
        self.create_results_tab()
        
        # 제어 버튼
        self.create_control_buttons()
        
    def create_header(self):
        """헤더 생성"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🚀 YOLOv7 Training GUI", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="Professional Object Detection Model Training Platform",
                                 font=('Arial', 12), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack()
    
    def create_settings_tab(self):
        """설정 탭 생성"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ 학습 설정")
        
        # 스크롤 가능한 프레임
        canvas = tk.Canvas(settings_frame)
        scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 설정 섹션들
        self.create_dataset_section(scrollable_frame)
        self.create_training_params_section(scrollable_frame)
        self.create_options_section(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_dataset_section(self, parent):
        """데이터셋 설정 섹션"""
        dataset_frame = ttk.LabelFrame(parent, text="📁 Dataset Configuration", padding=10)
        dataset_frame.pack(fill='x', pady=5, padx=10)
        
        # 데이터셋 경로
        ttk.Label(dataset_frame, text="Dataset Path (data.yaml):").pack(anchor='w')
        dataset_path_frame = ttk.Frame(dataset_frame)
        dataset_path_frame.pack(fill='x', pady=2)
        
        dataset_entry = ttk.Entry(dataset_path_frame, textvariable=self.dataset_path_var, width=50)
        dataset_entry.pack(side='left', fill='x', expand=True)
        
        ttk.Button(dataset_path_frame, text="Browse", 
                  command=self.browse_dataset).pack(side='right', padx=(5, 0))
        
        # 모델 설정
        ttk.Label(dataset_frame, text="Model Configuration:").pack(anchor='w', pady=(10, 0))
        model_combo = ttk.Combobox(dataset_frame, textvariable=self.model_config_var, width=50)
        model_combo['values'] = [
            "cfg/training/yolov7.yaml",
            "cfg/training/yolov7x.yaml", 
            "cfg/training/yolov7-tiny.yaml",
            "cfg/training/yolov7-w6.yaml"
        ]
        model_combo.pack(fill='x', pady=2)
        
        # 사전 훈련된 가중치
        ttk.Label(dataset_frame, text="Pretrained Weights (선택사항):").pack(anchor='w', pady=(5, 0))
        weights_frame = ttk.Frame(dataset_frame)
        weights_frame.pack(fill='x', pady=2)
        
        weights_entry = ttk.Entry(weights_frame, textvariable=self.weights_path_var, width=50)
        weights_entry.pack(side='left', fill='x', expand=True)
        
        ttk.Button(weights_frame, text="Browse", 
                  command=self.browse_weights).pack(side='right', padx=(5, 0))
        
        # 이미지 크기
        ttk.Label(dataset_frame, text="Image Size:").pack(anchor='w', pady=(5, 0))
        size_combo = ttk.Combobox(dataset_frame, textvariable=self.image_size_var, width=50)
        size_combo['values'] = ["640", "800", "1280", "512"]
        size_combo.pack(fill='x', pady=2)
    
    def create_training_params_section(self, parent):
        """훈련 파라미터 섹션"""
        params_frame = ttk.LabelFrame(parent, text="⚙️ Training Parameters", padding=10)
        params_frame.pack(fill='x', pady=5, padx=10)
        
        # 2열 레이아웃
        left_frame = ttk.Frame(params_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(params_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Epochs
        ttk.Label(left_frame, text="Epochs:").pack(anchor='w')
        epochs_frame = ttk.Frame(left_frame)
        epochs_frame.pack(fill='x', pady=2)
        
        epochs_scale = ttk.Scale(epochs_frame, from_=1, to=1000, variable=self.epochs_var,
                                orient='horizontal', command=self.update_epochs_label)
        epochs_scale.pack(side='left', fill='x', expand=True)
        
        self.epochs_label = ttk.Label(epochs_frame, text="300")
        self.epochs_label.pack(side='right', padx=(5, 0))
        
        # Batch Size
        ttk.Label(left_frame, text="Batch Size:").pack(anchor='w', pady=(5, 0))
        batch_frame = ttk.Frame(left_frame)
        batch_frame.pack(fill='x', pady=2)
        
        batch_scale = ttk.Scale(batch_frame, from_=1, to=64, variable=self.batch_size_var,
                               orient='horizontal', command=self.update_batch_size_label)
        batch_scale.pack(side='left', fill='x', expand=True)
        
        self.batch_size_label = ttk.Label(batch_frame, text="16")
        self.batch_size_label.pack(side='right', padx=(5, 0))
        
        # Learning Rate
        ttk.Label(right_frame, text="Learning Rate:").pack(anchor='w')
        lr_frame = ttk.Frame(right_frame)
        lr_frame.pack(fill='x', pady=2)
        
        lr_scale = ttk.Scale(lr_frame, from_=0.001, to=0.1, variable=self.learning_rate_var,
                            orient='horizontal', command=self.update_lr_label)
        lr_scale.pack(side='left', fill='x', expand=True)
        
        self.lr_label = ttk.Label(lr_frame, text="0.01")
        self.lr_label.pack(side='right', padx=(5, 0))
        
        # Workers
        ttk.Label(right_frame, text="Workers:").pack(anchor='w', pady=(5, 0))
        workers_frame = ttk.Frame(right_frame)
        workers_frame.pack(fill='x', pady=2)
        
        workers_scale = ttk.Scale(workers_frame, from_=0, to=16, variable=self.workers_var,
                                 orient='horizontal', command=self.update_workers_label)
        workers_scale.pack(side='left', fill='x', expand=True)
        
        self.workers_label = ttk.Label(workers_frame, text="8")
        self.workers_label.pack(side='right', padx=(5, 0))
        
        # Device
        ttk.Label(params_frame, text="Device:").pack(anchor='w', pady=(10, 0))
        device_combo = ttk.Combobox(params_frame, textvariable=self.device_var, width=30)
        device_combo['values'] = ["0", "0,1", "0,1,2,3", "cpu"]
        device_combo.pack(fill='x', pady=2)
    
    def create_options_section(self, parent):
        """훈련 옵션 섹션"""
        options_frame = ttk.LabelFrame(parent, text="🎯 Training Options", padding=10)
        options_frame.pack(fill='x', pady=5, padx=10)
        
        # 2열 레이아웃
        left_options = ttk.Frame(options_frame)
        left_options.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        right_options = ttk.Frame(options_frame)
        right_options.pack(side='right', fill='x', expand=True, padx=(5, 0))
        
        # 왼쪽 옵션들
        ttk.Checkbutton(left_options, text="Cache Images", variable=self.cache_images_var).pack(anchor='w')
        ttk.Checkbutton(left_options, text="Multi-Scale Training", variable=self.multi_scale_var).pack(anchor='w')
        ttk.Checkbutton(left_options, text="Image Weights", variable=self.image_weights_var).pack(anchor='w')
        
        # 오른쪽 옵션들
        ttk.Checkbutton(right_options, text="Rectangular Training", variable=self.rect_var).pack(anchor='w')
        ttk.Checkbutton(right_options, text="Adam Optimizer", variable=self.adam_var).pack(anchor='w')
        ttk.Checkbutton(right_options, text="Sync BatchNorm", variable=self.sync_bn_var).pack(anchor='w')
        
        # 출력 설정
        output_frame = ttk.LabelFrame(parent, text="💾 Output Configuration", padding=10)
        output_frame.pack(fill='x', pady=5, padx=10)
        
        ttk.Label(output_frame, text="Experiment Name:").pack(anchor='w')
        ttk.Entry(output_frame, textvariable=self.experiment_name_var, width=30).pack(fill='x', pady=2)
    
    def create_progress_tab(self):
        """진행사항 탭 생성"""
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="📊 진행사항")
        
        # 상태 표시
        status_frame = ttk.LabelFrame(progress_frame, text="📊 Training Status", padding=10)
        status_frame.pack(fill='x', pady=10, padx=10)
        
        # 상태 표시기
        status_indicator_frame = ttk.Frame(status_frame)
        status_indicator_frame.pack(fill='x', pady=5)
        
        self.status_canvas = tk.Canvas(status_indicator_frame, width=20, height=20)
        self.status_canvas.pack(side='left', padx=(0, 10))
        self.status_dot = self.status_canvas.create_oval(5, 5, 15, 15, fill='red', outline='')
        
        self.status_label = ttk.Label(status_indicator_frame, textvariable=self.status_text_var, font=('Arial', 12))
        self.status_label.pack(side='left')
        
        # 진행률 바
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=10)
        
        self.progress_label = ttk.Label(status_frame, text="0%", font=('Arial', 12, 'bold'))
        self.progress_label.pack()
        
        # 현재 메트릭 표시
        metrics_frame = ttk.LabelFrame(progress_frame, text="📈 Current Metrics", padding=10)
        metrics_frame.pack(fill='x', pady=10, padx=10)
        
        # 메트릭 그리드
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill='x')
        
        # 메트릭 라벨들
        self.create_metric_displays(metrics_grid)
        
        # 훈련 로그
        log_frame = ttk.LabelFrame(progress_frame, text="📝 Training Log", padding=10)
        log_frame.pack(fill='both', expand=True, pady=10, padx=10)
        
        # 로그 텍스트 위젯
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(log_container, bg='#2c3e50', fg='#ecf0f1', font=('Courier', 9),
                               height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")
        
        # 초기 로그 메시지
        self.add_log_entry("💡 YOLOv7 Professional GUI가 준비되었습니다!")
        self.add_log_entry("📁 YOLOv7 경로: " + str(self.trainer.yolo_original_dir))
        self.add_log_entry("🎯 완전한 기능을 갖춘 훈련 인터페이스입니다.")
        self.add_log_entry("⚙️ 설정 탭에서 파라미터를 조정하고 훈련을 시작하세요.")
    
    def create_metric_displays(self, parent):
        """메트릭 표시 위젯들 생성"""
        metrics = [
            ("Epoch", "current_epoch", "0/0"),
            ("Loss", "current_loss", "-"),
            ("Precision", "current_precision", "-"),
            ("Recall", "current_recall", "-"),
            ("mAP@0.5", "current_map50", "-"),
            ("mAP@0.5:0.95", "current_map95", "-")
        ]
        
        for i, (label_text, var_name, default_value) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            metric_frame = ttk.Frame(parent, relief='solid', borderwidth=1)
            metric_frame.grid(row=row, column=col, sticky='ew', padx=2, pady=2)
            
            ttk.Label(metric_frame, text=label_text, font=('Arial', 9, 'bold')).pack(pady=2)
            
            value_label = ttk.Label(metric_frame, text=default_value, font=('Arial', 12, 'bold'), 
                                   foreground='#3498db')
            value_label.pack(pady=2)
            
            # 참조 저장
            setattr(self, f"{var_name}_label", value_label)
        
        # 그리드 가중치
        for i in range(3):
            parent.grid_columnconfigure(i, weight=1)
    
    def create_results_tab(self):
        """결과 탭 생성"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📈 결과")
        
        # 차트 프레임
        charts_frame = ttk.LabelFrame(results_frame, text="📊 Performance Charts", padding=10)
        charts_frame.pack(fill='both', expand=True, pady=10, padx=10)
        
        # Matplotlib 차트
        self.create_charts(charts_frame)
    
    def create_charts(self, parent):
        """성능 차트 생성"""
        # Figure 생성
        self.fig = Figure(figsize=(12, 6))
        
        # 서브플롯들
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        # 차트 설정
        self.ax1.set_title("Precision & Recall")
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Score")
        
        self.ax2.set_title("mAP Metrics")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("mAP Score")
        
        self.ax3.set_title("Loss")
        self.ax3.set_xlabel("Epoch")
        self.ax3.set_ylabel("Loss")
        
        self.ax4.set_title("Learning Rate")
        self.ax4.set_xlabel("Epoch")
        self.ax4.set_ylabel("Learning Rate")
        
        self.fig.tight_layout()
        
        # 캔버스
        self.chart_canvas = FigureCanvasTkAgg(self.fig, parent)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 초기 데이터
        self.chart_data = {
            'epochs': [],
            'precision': [],
            'recall': [],
            'map50': [],
            'map95': [],
            'loss': [],
            'lr': []
        }
    
    def create_control_buttons(self):
        """제어 버튼들 생성"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        # 버튼들을 중앙에 정렬
        center_frame = ttk.Frame(button_frame)
        center_frame.pack(expand=True)
        
        self.start_btn = ttk.Button(center_frame, text="🚀 Start Training", 
                                   command=self.start_training)
        self.start_btn.pack(side='left', padx=5)
        
        self.pause_btn = ttk.Button(center_frame, text="⏸️ Pause", 
                                   command=self.pause_training, state='disabled')
        self.pause_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(center_frame, text="⏹️ Stop", 
                                  command=self.stop_training, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        self.reset_btn = ttk.Button(center_frame, text="🔄 Reset", 
                                   command=self.reset_settings)
        self.reset_btn.pack(side='left', padx=5)
        
        # 추가 기능 버튼들
        ttk.Button(center_frame, text="🧪 Test Connection", 
                  command=self.test_connection).pack(side='right', padx=5)
    
    def setup_callbacks(self):
        """YOLOv7 트레이너 콜백 설정"""
        self.trainer.register_callback('training_started', self.on_training_started)
        self.trainer.register_callback('metrics_update', self.on_metrics_update)
        self.trainer.register_callback('log_update', self.on_log_update)
        self.trainer.register_callback('training_complete', self.on_training_complete)
        self.trainer.register_callback('training_stopped', self.on_training_stopped)
        self.trainer.register_callback('error', self.on_error)
    
    # 이벤트 핸들러들
    def browse_dataset(self):
        """데이터셋 파일 선택"""
        filename = filedialog.askopenfilename(
            title="Select Dataset YAML File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if filename:
            self.dataset_path_var.set(filename)
            self.add_log_entry(f"📂 데이터셋 선택: {Path(filename).name}")
    
    def browse_weights(self):
        """가중치 파일 선택"""
        filename = filedialog.askopenfilename(
            title="Select Pretrained Weights",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.weights_path_var.set(filename)
            self.add_log_entry(f"📂 가중치 선택: {Path(filename).name}")
    
    # 스케일 업데이트 메서드들
    def update_epochs_label(self, value):
        self.epochs_label.config(text=str(int(float(value))))
    
    def update_batch_size_label(self, value):
        self.batch_size_label.config(text=str(int(float(value))))
    
    def update_lr_label(self, value):
        self.lr_label.config(text=f"{float(value):.3f}")
    
    def update_workers_label(self, value):
        self.workers_label.config(text=str(int(float(value))))
    
    # 훈련 제어 메서드들
    def start_training(self):
        """훈련 시작"""
        if self.is_training:
            return
        
        # 설정 검증
        if not self.validate_settings():
            return
        
        # UI 설정을 YOLOv7 설정으로 변환
        ui_config = self.get_ui_config()
        training_config = self.config_manager.get_training_config(ui_config)
        
        try:
            # 훈련 시작
            self.trainer.start_training(training_config)
            
            # 진행사항 탭으로 전환
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("훈련 시작 실패", str(e))
            self.add_log_entry(f"❌ 훈련 시작 실패: {e}")
    
    def pause_training(self):
        """훈련 일시정지"""
        if self.trainer.pause_training():
            self.add_log_entry("⏸️ 훈련이 일시정지되었습니다.")
    
    def stop_training(self):
        """훈련 정지"""
        if messagebox.askyesno("훈련 정지", "정말로 훈련을 정지하시겠습니까?"):
            if self.trainer.stop_training():
                self.add_log_entry("⏹️ 훈련이 정지되었습니다.")
    
    def reset_settings(self):
        """설정 초기화"""
        if messagebox.askyesno("설정 초기화", "모든 설정을 초기화하시겠습니까?"):
            # 모든 변수들을 기본값으로 재설정
            self.epochs_var.set(300)
            self.batch_size_var.set(16)
            self.learning_rate_var.set(0.01)
            self.workers_var.set(8)
            self.device_var.set("0")
            
            # 라벨 업데이트
            self.update_epochs_label(300)
            self.update_batch_size_label(16)
            self.update_lr_label(0.01)
            self.update_workers_label(8)
            
            # 경로 초기화
            self.dataset_path_var.set("")
            self.weights_path_var.set("")
            
            self.add_log_entry("🔄 설정이 초기화되었습니다.")
    
    def test_connection(self):
        """연결 테스트"""
        try:
            test_config = {
                'dataset_path': 'test.yaml',
                'model_config': 'cfg/training/yolov7.yaml',
                'epochs': 1,
                'batch_size': 1,
                'image_size': 640,
                'device': 'cpu',
                'experiment_name': 'connection_test'
            }
            
            yolo_config = self.config_manager.get_training_config(test_config)
            cmd = self.trainer.build_command(yolo_config)
            
            self.add_log_entry("🧪 연결 테스트 성공!")
            self.add_log_entry(f"🔧 생성된 명령어: {len(cmd)} 인자")
            messagebox.showinfo("테스트 성공", "YOLOv7 연결이 정상적으로 작동합니다!")
            
        except Exception as e:
            self.add_log_entry(f"❌ 연결 테스트 실패: {e}")
            messagebox.showerror("테스트 실패", f"연결 오류: {e}")
    
    def validate_settings(self):
        """설정 유효성 검사"""
        if not self.dataset_path_var.get():
            messagebox.showerror("설정 오류", "데이터셋 경로를 선택해주세요.")
            return False
        
        dataset_path = Path(self.dataset_path_var.get())
        if not dataset_path.exists():
            messagebox.showerror("설정 오류", f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
            return False
        
        return True
    
    def get_ui_config(self):
        """UI 설정을 딕셔너리로 반환"""
        return {
            'dataset_path': self.dataset_path_var.get(),
            'model_config': self.model_config_var.get(),
            'weights_path': self.weights_path_var.get(),
            'epochs': self.epochs_var.get(),
            'batch_size': self.batch_size_var.get(),
            'image_size': int(self.image_size_var.get()),
            'device': self.device_var.get(),
            'workers': self.workers_var.get(),
            'learning_rate': self.learning_rate_var.get(),
            'experiment_name': self.experiment_name_var.get(),
            
            # 옵션들
            'cache_images': self.cache_images_var.get(),
            'multi_scale': self.multi_scale_var.get(),
            'image_weights': self.image_weights_var.get(),
            'rect': self.rect_var.get(),
            'adam': self.adam_var.get(),
            'sync_bn': self.sync_bn_var.get(),
        }
    
    # 콜백 메서드들
    def on_training_started(self, data):
        """훈련 시작 콜백"""
        self.is_training = True
        self.status_text_var.set("훈련 진행 중...")
        self.status_canvas.itemconfig(self.status_dot, fill='green')
        
        # 버튼 상태 변경
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')
        
        self.add_log_entry("🚀 YOLOv7 훈련이 시작되었습니다!")
        config = data.get('config', {})
        self.add_log_entry(f"📊 Epochs: {config.get('epochs', 'N/A')}, Batch Size: {config.get('batch_size', 'N/A')}")
        self.add_log_entry(f"🎯 Dataset: {config.get('dataset_path', 'N/A')}")
    
    def on_metrics_update(self, metrics):
        """메트릭 업데이트 콜백"""
        self.current_metrics.update(metrics)
        
        # UI 업데이트 (메인 스레드에서 실행)
        self.root.after(0, self.update_metrics_display, metrics)
    
    def on_log_update(self, data):
        """로그 업데이트 콜백"""
        log_line = data.get('line', '')
        self.root.after(0, self.add_log_entry, log_line)
    
    def on_training_complete(self, data):
        """훈련 완료 콜백"""
        self.is_training = False
        success = data.get('success', False)
        
        if success:
            self.status_text_var.set("훈련 완료!")
            self.status_canvas.itemconfig(self.status_dot, fill='blue')
            self.add_log_entry("🎉 훈련이 성공적으로 완료되었습니다!")
            
            # 결과 탭으로 전환
            self.notebook.select(2)
        else:
            self.status_text_var.set("훈련 실패")
            self.status_canvas.itemconfig(self.status_dot, fill='red')
            return_code = data.get('return_code', 'Unknown')
            self.add_log_entry(f"❌ 훈련이 실패했습니다. 종료 코드: {return_code}")
        
        # 버튼 상태 복원
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
    
    def on_training_stopped(self, data):
        """훈련 정지 콜백"""
        self.is_training = False
        self.status_text_var.set("훈련 정지됨")
        self.status_canvas.itemconfig(self.status_dot, fill='red')
        
        # 버튼 상태 복원
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
    
    def on_error(self, data):
        """에러 콜백"""
        error_message = data.get('message', 'Unknown error')
        self.add_log_entry(f"❌ 오류: {error_message}")
        messagebox.showerror("훈련 오류", error_message)
    
    def update_metrics_display(self, metrics):
        """메트릭 표시 업데이트"""
        # Epoch 정보
        if 'current_epoch' in metrics and 'total_epochs' in metrics:
            epoch_text = f"{metrics['current_epoch']}/{metrics['total_epochs']}"
            self.current_epoch_label.config(text=epoch_text)
            
            # 진행률 계산
            progress = (metrics['current_epoch'] / metrics['total_epochs']) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{progress:.1f}%")
        
        # 메트릭 업데이트
        if 'precision' in metrics:
            self.current_precision_label.config(text=f"{metrics['precision']:.3f}")
        
        if 'recall' in metrics:
            self.current_recall_label.config(text=f"{metrics['recall']:.3f}")
        
        if 'map50' in metrics:
            self.current_map50_label.config(text=f"{metrics['map50']:.3f}")
        
        if 'map95' in metrics:
            self.current_map95_label.config(text=f"{metrics['map95']:.3f}")
        
        if 'loss' in metrics:
            self.current_loss_label.config(text=f"{metrics['loss']:.4f}")
        
        # 차트 데이터 업데이트
        self.update_charts_data(metrics)
    
    def update_charts_data(self, metrics):
        """차트 데이터 업데이트"""
        if 'current_epoch' in metrics:
            epoch = metrics['current_epoch']
            
            # 새로운 데이터 추가
            if epoch not in self.chart_data['epochs']:
                self.chart_data['epochs'].append(epoch)
                
                # 메트릭 데이터 추가
                self.chart_data['precision'].append(metrics.get('precision', 0))
                self.chart_data['recall'].append(metrics.get('recall', 0))
                self.chart_data['map50'].append(metrics.get('map50', 0))
                self.chart_data['map95'].append(metrics.get('map95', 0))
                self.chart_data['loss'].append(metrics.get('loss', 0))
                self.chart_data['lr'].append(metrics.get('learning_rate', 0))
                
                # 차트 업데이트
                self.update_charts()
    
    def update_charts(self):
        """차트 업데이트"""
        if len(self.chart_data['epochs']) < 2:
            return
        
        # 차트 지우기
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        epochs = self.chart_data['epochs']
        
        # Precision & Recall 차트
        if self.chart_data['precision'] and self.chart_data['recall']:
            self.ax1.plot(epochs, self.chart_data['precision'], 'b-', label='Precision', linewidth=2)
            self.ax1.plot(epochs, self.chart_data['recall'], 'r-', label='Recall', linewidth=2)
            self.ax1.set_title("Precision & Recall")
            self.ax1.set_xlabel("Epoch")
            self.ax1.set_ylabel("Score")
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
        
        # mAP 차트
        if self.chart_data['map50'] and self.chart_data['map95']:
            self.ax2.plot(epochs, self.chart_data['map50'], 'g-', label='mAP@0.5', linewidth=2)
            self.ax2.plot(epochs, self.chart_data['map95'], 'purple', label='mAP@0.5:0.95', linewidth=2)
            self.ax2.set_title("mAP Metrics")
            self.ax2.set_xlabel("Epoch")
            self.ax2.set_ylabel("mAP Score")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
        
        # Loss 차트
        if self.chart_data['loss']:
            self.ax3.plot(epochs, self.chart_data['loss'], 'orange', linewidth=2)
            self.ax3.set_title("Training Loss")
            self.ax3.set_xlabel("Epoch")
            self.ax3.set_ylabel("Loss")
            self.ax3.grid(True, alpha=0.3)
        
        # Learning Rate 차트
        if self.chart_data['lr']:
            self.ax4.plot(epochs, self.chart_data['lr'], 'brown', linewidth=2)
            self.ax4.set_title("Learning Rate")
            self.ax4.set_xlabel("Epoch")
            self.ax4.set_ylabel("Learning Rate")
            self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.chart_canvas.draw()
    
    def add_log_entry(self, message):
        """로그 엔트리 추가"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\\n"
        
        # 텍스트 위젯에 추가
        self.log_text.insert(tk.END, log_message)
        
        # 자동 스크롤
        self.log_text.see(tk.END)
        
        # 로그 길이 제한 (1000줄)
        lines = self.log_text.get("1.0", tk.END).split('\\n')
        if len(lines) > 1000:
            # 처음 100줄 삭제
            self.log_text.delete("1.0", "101.0")
    
    def show(self):
        """윈도우 표시"""
        self.root.deiconify()  # 윈도우 숨김 해제
'''

    # 파일 작성
    with open("src/ui/main_window.py", 'w', encoding='utf-8') as f:
        f.write(full_ui_code)
    
    print("✅ 완전한 UI로 업데이트 완료!")

if __name__ == "__main__":
    if Path.cwd().name != "yolov7_gui_standalone":
        print("❌ yolov7_gui_standalone 폴더에서 실행하세요!")
        exit(1)
    
    update_to_full_ui()
    print("\n🎉 완전한 GUI 업데이트 완료!")
    print("📋 다음 단계: python main.py (완전한 GUI가 실행됩니다)")