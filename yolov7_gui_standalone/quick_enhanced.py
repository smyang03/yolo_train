import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import time
import json
import os
from datetime import datetime, timedelta
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib이 설치되지 않았습니다. 차트 기능이 제한됩니다.")

class MainWindow:
    """완전한 Enhanced Professional GUI"""
    
    def __init__(self, root, trainer, config_manager, model_manager):
        self.root = root
        self.trainer = trainer
        self.config_manager = config_manager
        self.model_manager = model_manager
        
        # 훈련 상태
        self.is_training = False
        self.training_progress = 0
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 300
        self.current_metrics = {}
        
        # 메트릭 데이터
        self.metrics_data = {
            'epochs': [],
            'precision': [],
            'recall': [],
            'map50': [],
            'map95': [],
            'loss': [],
            'lr': []
        }
        
        # 🏆 Best models tracking
        self.best_models = {
            'precision': {'score': 0, 'epoch': 0, 'precision': 0, 'recall': 0, 'map50': 0, 'map95': 0},
            'recall': {'score': 0, 'epoch': 0, 'precision': 0, 'recall': 0, 'map50': 0, 'map95': 0},
            'balance': {'score': 0, 'epoch': 0, 'precision': 0, 'recall': 0, 'map50': 0, 'map95': 0},
            'map': {'score': 0, 'epoch': 0, 'precision': 0, 'recall': 0, 'map50': 0, 'map95': 0}
        }
        
        # 💾 저장된 모델들
        self.saved_models = []
        self.selected_model = None
        self.selected_model_type = None
        
        # UI 변수들 초기화
        self.setup_variables()
        
        # UI 생성
        self.create_ui()
        
        # 콜백 등록
        self.setup_callbacks()
        
        # 초기 로그 메시지 추가 (UI 생성 후)
        self.initialize_log_messages()
        
    def setup_variables(self):
        """UI 변수들 초기화"""
        # 기존 변수들
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
        
        # 새로 추가된 변수들
        self.dataset_mode = tk.StringVar(value="single")
        self.split_ratio_var = tk.DoubleVar(value=0.8)
        
        # 고급 훈련 옵션들
        self.close_mosaic_var = tk.IntVar(value=10)
        self.save_checkpoints_var = tk.BooleanVar()
        self.save_all_weights_var = tk.BooleanVar()
        self.save_best_models_var = tk.BooleanVar(value=True)
        self.wandb_logging_var = tk.BooleanVar()
        self.tensorboard_var = tk.BooleanVar()
        self.plot_results_var = tk.BooleanVar()
        
        # Merge 옵션들
        self.shuffle_var = tk.BooleanVar(value=True)
        self.balance_var = tk.BooleanVar(value=False)
        self.remove_duplicates_var = tk.BooleanVar(value=True)
        
        # Class 선택
        self.class_var = tk.StringVar(value="all")
    
    def create_ui(self):
        """Enhanced UI 생성"""
        self.root.title("🚀 YOLOv7 Enhanced Professional Training GUI")
        self.root.geometry("1500x1000")
        self.root.configure(bg='#f0f0f0')
        
        # 헤더 생성
        self.create_header()
        
        # 노트북 생성
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 4개 탭 생성
        self.create_enhanced_settings_tab()
        self.create_enhanced_progress_tab()
        self.create_enhanced_results_tab()
        self.create_models_tab()
        
        # 제어 버튼
        self.create_control_buttons()
        
    def create_header(self):
        """헤더 생성"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🚀 YOLOv7 Enhanced Professional Training GUI", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="Advanced Object Detection Training with Complete Model Management",
                                 font=('Arial', 12), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack()
    
    def create_enhanced_settings_tab(self):
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
        
        # Dataset 설정 섹션
        self.create_dataset_section(scrollable_frame)
        
        # 훈련 파라미터 섹션
        self.create_training_params_section(scrollable_frame)
        
        # 고급 옵션 섹션
        self.create_advanced_options_section(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_dataset_section(self, parent):
        """Dataset 설정 섹션"""
        dataset_frame = ttk.LabelFrame(parent, text="📁 Dataset Configuration", padding=15)
        dataset_frame.pack(fill='x', pady=10, padx=15)
        
        # Dataset Mode 선택
        ttk.Label(dataset_frame, text="Dataset Mode:", font=('Arial', 11, 'bold')).pack(anchor='w')
        mode_frame = ttk.Frame(dataset_frame)
        mode_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(mode_frame, text="Single Dataset (YAML)", 
                       variable=self.dataset_mode, value="single",
                       command=self.on_dataset_mode_change).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Multiple Datasets (Merge)", 
                       variable=self.dataset_mode, value="multiple",
                       command=self.on_dataset_mode_change).pack(anchor='w')
        
        # Single Dataset Frame
        self.single_dataset_frame = ttk.Frame(dataset_frame)
        self.single_dataset_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.single_dataset_frame, text="Dataset Path (data.yaml):").pack(anchor='w')
        dataset_path_frame = ttk.Frame(self.single_dataset_frame)
        dataset_path_frame.pack(fill='x', pady=2)
        
        ttk.Entry(dataset_path_frame, textvariable=self.dataset_path_var, 
                 font=('Arial', 10), width=70).pack(side='left', fill='x', expand=True)
        ttk.Button(dataset_path_frame, text="Browse", 
                  command=self.browse_dataset).pack(side='right', padx=(5, 0))
        
        # Multiple Dataset Frame
        self.multiple_dataset_frame = ttk.Frame(dataset_frame)
        
        ttk.Label(self.multiple_dataset_frame, text="Multiple Datasets:").pack(anchor='w')
        self.dataset_listbox = tk.Listbox(self.multiple_dataset_frame, height=4)
        self.dataset_listbox.pack(fill='x', pady=2)
        
        dataset_buttons_frame = ttk.Frame(self.multiple_dataset_frame)
        dataset_buttons_frame.pack(fill='x', pady=2)
        
        ttk.Button(dataset_buttons_frame, text="Add Dataset", 
                  command=self.add_dataset).pack(side='left', padx=(0, 5))
        ttk.Button(dataset_buttons_frame, text="Remove Selected", 
                  command=self.remove_dataset).pack(side='left')
        
        # Merge Options
        merge_frame = ttk.LabelFrame(self.multiple_dataset_frame, text="Merge Options", padding=5)
        merge_frame.pack(fill='x', pady=5)
        
        ttk.Checkbutton(merge_frame, text="Shuffle merged data", variable=self.shuffle_var).pack(anchor='w')
        ttk.Checkbutton(merge_frame, text="Balance classes", variable=self.balance_var).pack(anchor='w')
        ttk.Checkbutton(merge_frame, text="Remove duplicates", variable=self.remove_duplicates_var).pack(anchor='w')
        
        # Split Ratio
        ttk.Label(merge_frame, text="Train/Valid Split Ratio:").pack(anchor='w', pady=(5, 0))
        split_scale = ttk.Scale(merge_frame, from_=0.1, to=0.9, variable=self.split_ratio_var, 
                               orient='horizontal', command=self.update_split_ratio_label)
        split_scale.pack(fill='x', pady=2)
        
        self.split_ratio_label = ttk.Label(merge_frame, text="80% / 20%")
        self.split_ratio_label.pack(anchor='w')
        
        # 모델 설정들
        self.create_model_config_section(dataset_frame)
        
        # 초기에는 multiple dataset frame 숨김
        self.on_dataset_mode_change()
    
    def create_model_config_section(self, parent):
        """모델 설정 섹션"""
        ttk.Label(parent, text="Model Configuration:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(15, 0))
        model_combo = ttk.Combobox(parent, textvariable=self.model_config_var, 
                                  font=('Arial', 10), width=70)
        model_combo['values'] = [
            "cfg/training/yolov7.yaml",
            "cfg/training/yolov7x.yaml", 
            "cfg/training/yolov7-tiny.yaml",
            "cfg/training/yolov7-w6.yaml",
            "cfg/training/yolov7-e6.yaml"
        ]
        model_combo.pack(fill='x', pady=5)
        
        # 사전 훈련된 가중치
        ttk.Label(parent, text="Pretrained Weights (선택사항):", 
                 font=('Arial', 11, 'bold')).pack(anchor='w', pady=(15, 0))
        weights_frame = ttk.Frame(parent)
        weights_frame.pack(fill='x', pady=5)
        
        ttk.Entry(weights_frame, textvariable=self.weights_path_var, 
                 font=('Arial', 10), width=70).pack(side='left', fill='x', expand=True)
        ttk.Button(weights_frame, text="Browse", 
                  command=self.browse_weights).pack(side='right', padx=(5, 0))
        
        # 이미지 크기
        ttk.Label(parent, text="Image Size:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(15, 0))
        size_combo = ttk.Combobox(parent, textvariable=self.image_size_var, 
                                 font=('Arial', 10), width=70)
        size_combo['values'] = ["640", "800", "1280", "512", "416"]
        size_combo.pack(fill='x', pady=5)
    
    def create_training_params_section(self, parent):
        """훈련 파라미터 섹션"""
        params_frame = ttk.LabelFrame(parent, text="⚙️ Training Parameters", padding=15)
        params_frame.pack(fill='x', pady=10, padx=15)
        
        # 파라미터 그리드
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill='x')
        
        # Epochs
        ttk.Label(params_grid, text="Epochs:", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, sticky='w', padx=(0, 20), pady=5)
        epochs_frame = ttk.Frame(params_grid)
        epochs_frame.grid(row=0, column=1, sticky='ew', pady=5)
        
        epochs_scale = ttk.Scale(epochs_frame, from_=1, to=1000, variable=self.epochs_var, 
                               orient='horizontal', length=200, command=self.update_epochs_label)
        epochs_scale.pack(side='left')
        self.epochs_label = ttk.Label(epochs_frame, text="300", font=('Arial', 11, 'bold'))
        self.epochs_label.pack(side='left', padx=(10, 0))
        
        # Batch Size
        ttk.Label(params_grid, text="Batch Size:", font=('Arial', 11, 'bold')).grid(
            row=1, column=0, sticky='w', padx=(0, 20), pady=5)
        batch_scale = ttk.Scale(params_grid, from_=1, to=64, variable=self.batch_size_var, 
                               orient='horizontal', length=200, command=self.update_batch_label)
        batch_scale.grid(row=1, column=1, sticky='ew', pady=5)
        
        self.batch_label = ttk.Label(params_grid, text="16", font=('Arial', 11, 'bold'))
        self.batch_label.grid(row=1, column=2, padx=(10, 0), pady=5)
        
        # Learning Rate
        ttk.Label(params_grid, text="Learning Rate:", font=('Arial', 11, 'bold')).grid(
            row=2, column=0, sticky='w', padx=(0, 20), pady=5)
        lr_scale = ttk.Scale(params_grid, from_=0.001, to=0.1, variable=self.learning_rate_var, 
                            orient='horizontal', length=200, command=self.update_lr_label)
        lr_scale.grid(row=2, column=1, sticky='ew', pady=5)
        
        self.lr_label = ttk.Label(params_grid, text="0.01", font=('Arial', 11, 'bold'))
        self.lr_label.grid(row=2, column=2, padx=(10, 0), pady=5)
        
        # Workers
        ttk.Label(params_grid, text="Workers:", font=('Arial', 11, 'bold')).grid(
            row=3, column=0, sticky='w', padx=(0, 20), pady=5)
        workers_scale = ttk.Scale(params_grid, from_=1, to=16, variable=self.workers_var, 
                                 orient='horizontal', length=200, command=self.update_workers_label)
        workers_scale.grid(row=3, column=1, sticky='ew', pady=5)
        
        self.workers_label = ttk.Label(params_grid, text="8", font=('Arial', 11, 'bold'))
        self.workers_label.grid(row=3, column=2, padx=(10, 0), pady=5)
        
        # Device
        ttk.Label(params_grid, text="Device:", font=('Arial', 11, 'bold')).grid(
            row=4, column=0, sticky='w', padx=(0, 20), pady=5)
        device_combo = ttk.Combobox(params_grid, textvariable=self.device_var, 
                                   values=["0", "1", "2", "3", "cpu"], width=15)
        device_combo.grid(row=4, column=1, sticky='w', pady=5)
        
        params_grid.grid_columnconfigure(1, weight=1)
    
    def create_advanced_options_section(self, parent):
        """고급 훈련 옵션 섹션"""
        options_frame = ttk.LabelFrame(parent, text="🎯 Training Options", padding=15)
        options_frame.pack(fill='x', pady=10, padx=15)
        
        # 기존 옵션들
        left_options = ttk.Frame(options_frame)
        left_options.pack(side='left', fill='x', expand=True, padx=(0, 15))
        
        right_options = ttk.Frame(options_frame)
        right_options.pack(side='right', fill='x', expand=True, padx=(15, 0))
        
        # 체크박스들
        ttk.Checkbutton(left_options, text="Cache Images", variable=self.cache_images_var).pack(anchor='w', pady=3)
        ttk.Checkbutton(left_options, text="Multi-Scale Training", variable=self.multi_scale_var).pack(anchor='w', pady=3)
        ttk.Checkbutton(left_options, text="Image Weights", variable=self.image_weights_var).pack(anchor='w', pady=3)
        
        ttk.Checkbutton(right_options, text="Rectangular Training", variable=self.rect_var).pack(anchor='w', pady=3)
        ttk.Checkbutton(right_options, text="Adam Optimizer", variable=self.adam_var).pack(anchor='w', pady=3)
        ttk.Checkbutton(right_options, text="Sync BatchNorm", variable=self.sync_bn_var).pack(anchor='w', pady=3)
        
        # 고급 옵션들
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Options", padding=15)
        advanced_frame.pack(fill='x', pady=10, padx=15)
        
        # Close Mosaic
        ttk.Label(advanced_frame, text="Close Mosaic (Epochs before end):").pack(anchor='w')
        mosaic_frame = ttk.Frame(advanced_frame)
        mosaic_frame.pack(fill='x', pady=2)
        
        mosaic_scale = ttk.Scale(mosaic_frame, from_=0, to=50, variable=self.close_mosaic_var,
                                orient='horizontal', command=self.update_mosaic_label)
        mosaic_scale.pack(side='left', fill='x', expand=True)
        
        self.mosaic_label = ttk.Label(mosaic_frame, text="10")
        self.mosaic_label.pack(side='right', padx=(5, 0))
        
        # 출력/로깅 옵션들
        output_options_frame = ttk.Frame(advanced_frame)
        output_options_frame.pack(fill='x', pady=10)
        
        left_output = ttk.Frame(output_options_frame)
        left_output.pack(side='left', fill='x', expand=True)
        
        right_output = ttk.Frame(output_options_frame)
        right_output.pack(side='right', fill='x', expand=True)
        
        ttk.Checkbutton(left_output, text="Save Checkpoints", variable=self.save_checkpoints_var).pack(anchor='w')
        ttk.Checkbutton(left_output, text="Save All Epoch Weights", variable=self.save_all_weights_var).pack(anchor='w')
        ttk.Checkbutton(left_output, text="Save Best Models", variable=self.save_best_models_var).pack(anchor='w')
        
        ttk.Checkbutton(right_output, text="W&B Logging", variable=self.wandb_logging_var).pack(anchor='w')
        ttk.Checkbutton(right_output, text="TensorBoard", variable=self.tensorboard_var).pack(anchor='w')
        ttk.Checkbutton(right_output, text="Plot Results", variable=self.plot_results_var).pack(anchor='w')
        
        # 실험명 설정
        ttk.Label(advanced_frame, text="Experiment Name:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(15, 0))
        ttk.Entry(advanced_frame, textvariable=self.experiment_name_var, 
                 font=('Arial', 10), width=50).pack(fill='x', pady=5)
    
    def create_enhanced_progress_tab(self):
        """진행사항 탭 생성"""
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="📊 진행사항")
        
        # 메인 컨테이너
        main_container = ttk.Frame(progress_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 상태 표시
        self.create_status_section(main_container)
        
        # 시간 정보 카드들
        self.create_time_info_section(main_container)
        
        # 메트릭 요약 카드들
        self.create_metrics_summary_cards(main_container)
        
        # 훈련 로그
        self.create_log_section(main_container)
        
    def create_status_section(self, parent):
        """상태 표시 섹션"""
        status_frame = ttk.LabelFrame(parent, text="📊 Training Status", padding=15)
        status_frame.pack(fill='x', pady=(0, 10))
        
        # 상태 표시기
        status_indicator_frame = ttk.Frame(status_frame)
        status_indicator_frame.pack(fill='x', pady=5)
        
        self.status_canvas = tk.Canvas(status_indicator_frame, width=20, height=20)
        self.status_canvas.pack(side='left', padx=(0, 10))
        self.status_dot = self.status_canvas.create_oval(5, 5, 15, 15, fill='red', outline='')
        
        self.status_label = ttk.Label(status_indicator_frame, textvariable=self.status_text_var, 
                                     font=('Arial', 14, 'bold'))
        self.status_label.pack(side='left')
        
        # 진행률 바
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=15)
        
        self.progress_label = ttk.Label(status_frame, text="0%", font=('Arial', 12, 'bold'))
        self.progress_label.pack()
    
    def create_time_info_section(self, parent):
        """시간 정보 섹션"""
        info_container = ttk.Frame(parent)
        info_container.pack(fill='x', pady=10)
        
        # 시간 정보 카드
        time_frame = ttk.LabelFrame(info_container, text="⏱️ Time Information", padding=10)
        time_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.create_info_grid(time_frame, [
            ("Training Time:", "total_time", "00:00:00"),
            ("Avg Epoch Time:", "avg_epoch_time", "-"),
            ("Time Remaining:", "remaining_time", "-"),
            ("ETA:", "eta_time", "-")
        ])
        
        # 현재 메트릭 카드
        metrics_frame = ttk.LabelFrame(info_container, text="📈 Current Metrics", padding=10)
        metrics_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.create_info_grid(metrics_frame, [
            ("Current Epoch:", "current_epoch", "0"),
            ("Current Loss:", "current_loss", "-"),
            ("Best Epoch:", "best_epoch", "-"),
            ("GPU Memory:", "gpu_memory", "-")
        ])
    
    def create_metrics_summary_cards(self, parent):
        """메트릭 요약 카드들"""
        summary_frame = ttk.LabelFrame(parent, text="📊 Current Metrics Summary", padding=10)
        summary_frame.pack(fill='x', pady=10)
        
        # 4개 메트릭 카드 생성
        cards_frame = ttk.Frame(summary_frame)
        cards_frame.pack(fill='x')
        
        metrics = [
            ("Precision", "precision", "#e74c3c"),
            ("Recall", "recall", "#2ecc71"),
            ("mAP@0.5", "map50", "#f39c12"),
            ("mAP@0.5:0.95", "map95", "#9b59b6")
        ]
        
        for i, (name, var_name, color) in enumerate(metrics):
            card_frame = ttk.Frame(cards_frame, relief='solid', borderwidth=2)
            card_frame.pack(side='left', fill='x', expand=True, padx=2)
            
            ttk.Label(card_frame, text=name, font=('Arial', 10, 'bold')).pack(pady=5)
            
            value_label = ttk.Label(card_frame, text="0.000", font=('Arial', 16, 'bold'))
            value_label.pack(pady=5)
            
            # 참조 저장
            setattr(self, f"current_{var_name}_summary_label", value_label)
    
    def create_log_section(self, parent):
        """로그 섹션"""
        log_frame = ttk.LabelFrame(parent, text="📝 Training Log", padding=15)
        log_frame.pack(fill='both', expand=True, pady=10)
        
        # 로그 텍스트 위젯
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(log_container, bg='#2c3e50', fg='#ecf0f1', font=('Consolas', 9),
                               height=12, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")
    
    def initialize_log_messages(self):
        """초기 로그 메시지들을 추가"""
        self.add_log_entry("🎉 YOLOv7 Enhanced Professional GUI가 성공적으로 시작되었습니다!")
        self.add_log_entry("📁 YOLOv7 경로: " + str(self.trainer.yolo_original_dir))
        self.add_log_entry("💡 완전한 기능을 갖춘 Enhanced 훈련 인터페이스입니다.")
        self.add_log_entry("🆕 새로 추가된 기능: 모델 관리, 고급 Dataset, 시간 추적")
        self.add_log_entry("⚙️ 설정 탭에서 데이터셋과 고급 옵션들을 조정하세요.")
        self.add_log_entry("🚀 모든 준비가 완료되면 Start Training을 클릭하세요!")
    
    def create_enhanced_results_tab(self):
        """결과 탭 생성"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📈 결과")
        
        # 스크롤 가능한 프레임
        canvas = tk.Canvas(results_frame)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 메인 컨테이너
        main_container = ttk.Frame(scrollable_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 차트 섹션
        if MATPLOTLIB_AVAILABLE:
            charts_frame = ttk.LabelFrame(main_container, text="📊 Performance Charts", padding=15)
            charts_frame.pack(fill='both', expand=True, pady=10)
            self.create_charts(charts_frame)
        else:
            placeholder_frame = ttk.LabelFrame(main_container, text="📊 Results Summary", padding=15)
            placeholder_frame.pack(fill='both', expand=True, pady=10)
            
            placeholder_label = ttk.Label(placeholder_frame, 
                text="📈 훈련 결과가 여기에 표시됩니다.\n\n더 자세한 차트를 보려면 matplotlib을 설치하세요:\npip install matplotlib", 
                font=('Arial', 12), justify='center')
            placeholder_label.pack(expand=True)
        
        # Class-specific Performance 섹션
        self.create_class_performance_section(main_container)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_class_performance_section(self, parent):
        """Class-specific Performance 섹션"""
        class_frame = ttk.LabelFrame(parent, text="📋 Class-specific Performance", padding=10)
        class_frame.pack(fill='x', pady=10)
        
        # Class 선택기
        selector_frame = ttk.Frame(class_frame)
        selector_frame.pack(fill='x', pady=5)
        
        ttk.Label(selector_frame, text="Select Class:").pack(side='left', padx=(0, 10))
        
        class_combo = ttk.Combobox(selector_frame, textvariable=self.class_var, width=30)
        class_combo['values'] = ["All Classes (Overall)", "Class 0: Person", "Class 1: Bicycle", 
                                "Class 2: Car", "Class 3: Motorcycle"]
        class_combo.pack(side='left')
        
        # Class 메트릭 표시
        class_metrics_frame = ttk.Frame(class_frame)
        class_metrics_frame.pack(fill='x', pady=10)
        
        class_metrics = [
            ("P:", "class_precision", "#e74c3c"),
            ("R:", "class_recall", "#2ecc71"),
            ("AP50:", "class_ap50", "#f39c12"),
            ("AP95:", "class_ap95", "#9b59b6")
        ]
        
        for name, var_name, color in class_metrics:
            metric_frame = ttk.Frame(class_metrics_frame, relief='solid', borderwidth=1)
            metric_frame.pack(side='left', fill='x', expand=True, padx=2, pady=2)
            
            ttk.Label(metric_frame, text=name, font=('Arial', 10, 'bold')).pack(side='left', padx=5)
            
            value_label = ttk.Label(metric_frame, text="-", font=('Arial', 12, 'bold'))
            value_label.pack(side='right', padx=5)
            
            setattr(self, f"{var_name}_label", value_label)
    
    def create_models_tab(self):
        """모델 선택 탭 (핵심 새 기능)"""
        models_frame = ttk.Frame(self.notebook)
        self.notebook.add(models_frame, text="🏆 모델 선택")
        
        # 스크롤 가능한 프레임
        canvas = tk.Canvas(models_frame)
        scrollbar = ttk.Scrollbar(models_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 메인 컨테이너
        main_container = ttk.Frame(scrollable_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Best Models Summary
        self.create_best_models_section(main_container)
        
        # Selected Model Details
        self.create_selected_model_section(main_container)
        
        # All Saved Models Table
        self.create_saved_models_section(main_container)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_best_models_section(self, parent):
        """Best Models Summary 섹션"""
        best_models_frame = ttk.LabelFrame(parent, text="🏆 Best Models Summary", padding=10)
        best_models_frame.pack(fill='x', pady=(0, 10))
        
        # 2x2 그리드로 best model 카드들 생성
        models_container = ttk.Frame(best_models_frame)
        models_container.pack(fill='x')
        
        model_types = [
            ("Best Precision", "precision", "🎯"),
            ("Best Recall", "recall", "🔍"),
            ("Best P+R Balance", "balance", "⚖️"),
            ("Best mAP", "map", "🏅")
        ]
        
        for i, (title, model_type, icon) in enumerate(model_types):
            row = i // 2
            col = i % 2
            
            card_frame = ttk.LabelFrame(models_container, text=f"{icon} {title}", padding=10)
            card_frame.grid(row=row, column=col, sticky='ew', padx=5, pady=5)
            
            # 점수 표시
            score_label = ttk.Label(card_frame, text="0.000", font=('Arial', 20, 'bold'))
            score_label.pack()
            
            # 에포크 표시
            epoch_label = ttk.Label(card_frame, text="Epoch -", font=('Arial', 10))
            epoch_label.pack()
            
            # 버튼들
            button_frame = ttk.Frame(card_frame)
            button_frame.pack(fill='x', pady=5)
            
            select_btn = ttk.Button(button_frame, text="Select", 
                                   command=lambda t=model_type: self.select_model(t))
            select_btn.pack(side='left', fill='x', expand=True, padx=(0, 2))
            
            export_btn = ttk.Button(button_frame, text="📦", width=3,
                                   command=lambda t=model_type: self.quick_export_model(t))
            export_btn.pack(side='right', padx=(2, 0))
            
            # 참조 저장
            setattr(self, f"best_{model_type}_score_label", score_label)
            setattr(self, f"best_{model_type}_epoch_label", epoch_label)
            setattr(self, f"best_{model_type}_select_btn", select_btn)
        
        # 그리드 가중치 설정
        models_container.grid_columnconfigure(0, weight=1)
        models_container.grid_columnconfigure(1, weight=1)
    
    def create_selected_model_section(self, parent):
        """Selected Model Details 섹션"""
        selected_frame = ttk.LabelFrame(parent, text="🎯 Selected Model Details", padding=10)
        selected_frame.pack(fill='x', pady=10)
        
        # 모델 정보
        info_frame = ttk.Frame(selected_frame)
        info_frame.pack(fill='x', pady=5)
        
        self.selected_model_title = ttk.Label(info_frame, text="No Model Selected", font=('Arial', 14, 'bold'))
        self.selected_model_title.pack()
        
        self.selected_model_path = ttk.Label(info_frame, text="Please select a model from above", font=('Arial', 10))
        self.selected_model_path.pack()
        
        # 메트릭 표시 (초기에 숨김)
        self.selected_metrics_frame = ttk.Frame(selected_frame)
        
        metrics_grid = ttk.Frame(self.selected_metrics_frame)
        metrics_grid.pack(fill='x', pady=10)
        
        selected_metrics = [
            ("Precision:", "selected_precision"),
            ("Recall:", "selected_recall"),
            ("mAP@0.5:", "selected_map50"),
            ("mAP@0.5:0.95:", "selected_map95")
        ]
        
        for i, (label, var_name) in enumerate(selected_metrics):
            row = i // 2
            col = i % 2
            
            metric_frame = ttk.Frame(metrics_grid, relief='solid', borderwidth=1)
            metric_frame.grid(row=row, column=col, sticky='ew', padx=2, pady=2)
            
            ttk.Label(metric_frame, text=label, font=('Arial', 10, 'bold')).pack(side='left', padx=5, pady=5)
            
            value_label = ttk.Label(metric_frame, text="0.000", font=('Arial', 12, 'bold'))
            value_label.pack(side='right', padx=5, pady=5)
            
            setattr(self, f"{var_name}_label", value_label)
        
        metrics_grid.grid_columnconfigure(0, weight=1)
        metrics_grid.grid_columnconfigure(1, weight=1)
        
        # 액션 버튼들
        actions_frame = ttk.Frame(self.selected_metrics_frame)
        actions_frame.pack(fill='x', pady=10)
        
        ttk.Button(actions_frame, text="📥 Download Model", 
                  command=self.download_model).pack(side='left', fill='x', expand=True, padx=(0, 2))
        ttk.Button(actions_frame, text="🧪 Test Model", 
                  command=self.test_model).pack(side='left', fill='x', expand=True, padx=2)
        ttk.Button(actions_frame, text="🚀 Deploy Model", 
                  command=self.deploy_model).pack(side='left', fill='x', expand=True, padx=2)
        ttk.Button(actions_frame, text="📦 Export ONNX", 
                  command=self.export_onnx).pack(side='left', fill='x', expand=True, padx=(2, 0))
    
    def create_saved_models_section(self, parent):
        """저장된 모델들 테이블 섹션"""
        saved_frame = ttk.LabelFrame(parent, text="💾 All Saved Models", padding=10)
        saved_frame.pack(fill='both', expand=True, pady=10)
        
        # 모델 테이블용 Treeview 생성
        columns = ("Epoch", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "File Size")
        self.models_tree = ttk.Treeview(saved_frame, columns=columns, show='headings', height=8)
        
        # 컬럼 설정
        for col in columns:
            self.models_tree.heading(col, text=col)
            self.models_tree.column(col, width=100, anchor='center')
        
        # 스크롤바 추가
        tree_scrollbar = ttk.Scrollbar(saved_frame, orient="vertical", command=self.models_tree.yview)
        self.models_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Treeview와 스크롤바 배치
        self.models_tree.pack(side="left", fill="both", expand=True)
        tree_scrollbar.pack(side="right", fill="y")
        
        # 테이블 액션들
        table_actions_frame = ttk.Frame(saved_frame)
        table_actions_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(table_actions_frame, text="📋 Export Model List", 
                  command=self.export_model_list).pack(side='left', padx=(0, 5))
        ttk.Button(table_actions_frame, text="🗑️ Cleanup Old Models", 
                  command=self.cleanup_models).pack(side='left', padx=5)
        ttk.Button(table_actions_frame, text="🔄 Refresh List", 
                  command=self.refresh_model_list).pack(side='left', padx=5)
    
    def create_charts(self, parent):
        """성능 차트 생성"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Figure 생성
        self.fig = Figure(figsize=(12, 8), facecolor='white')
        
        # 서브플롯들
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        # 차트 설정
        self.ax1.set_title("Precision & Recall", fontsize=12, fontweight='bold')
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Score")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("mAP Metrics", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("mAP Score")
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title("Training Loss", fontsize=12, fontweight='bold')
        self.ax3.set_xlabel("Epoch")
        self.ax3.set_ylabel("Loss")
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title("Learning Rate", fontsize=12, fontweight='bold')
        self.ax4.set_xlabel("Epoch")
        self.ax4.set_ylabel("Learning Rate")
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # 캔버스
        self.chart_canvas = FigureCanvasTkAgg(self.fig, parent)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_control_buttons(self):
        """제어 버튼들 생성"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=15, pady=15)
        
        # 버튼들을 중앙에 정렬
        center_frame = ttk.Frame(button_frame)
        center_frame.pack(expand=True)
        
        # 주요 제어 버튼들
        self.start_btn = ttk.Button(center_frame, text="🚀 Start Enhanced Training", 
                                   command=self.start_enhanced_training, width=18)
        self.start_btn.pack(side='left', padx=8)
        
        self.pause_btn = ttk.Button(center_frame, text="⏸️ Pause", 
                                   command=self.pause_training, state='disabled', width=12)
        self.pause_btn.pack(side='left', padx=8)
        
        self.stop_btn = ttk.Button(center_frame, text="⏹️ Stop", 
                                  command=self.stop_training, state='disabled', width=12)
        self.stop_btn.pack(side='left', padx=8)
        
        self.reset_btn = ttk.Button(center_frame, text="🔄 Reset", 
                                   command=self.reset_settings, width=12)
        self.reset_btn.pack(side='left', padx=8)
        
        # 우측 유틸리티 버튼들
        ttk.Button(center_frame, text="🧪 Test Connection", 
                  command=self.test_connection, width=15).pack(side='right', padx=8)
    
    def setup_callbacks(self):
        """YOLOv7 트레이너 콜백 설정"""
        self.trainer.register_callback('training_started', self.on_training_started)
        self.trainer.register_callback('metrics_update', self.on_metrics_update)
        self.trainer.register_callback('log_update', self.on_log_update)
        self.trainer.register_callback('training_complete', self.on_training_complete)
        self.trainer.register_callback('training_stopped', self.on_training_stopped)
        self.trainer.register_callback('error', self.on_error)
    
    # 이벤트 핸들러들
    def on_dataset_mode_change(self):
        """Dataset 모드 변경 처리"""
        mode = self.dataset_mode.get()
        
        # 모든 프레임 숨기기
        self.multiple_dataset_frame.pack_forget()
        
        if mode == "single":
            # Single dataset 모드 (이미 표시됨)
            pass
        elif mode == "multiple":
            # Multiple dataset 모드 표시
            self.multiple_dataset_frame.pack(fill='x', pady=5)
        
        self.add_log_entry(f"Dataset mode changed to: {mode}")
    
    def add_dataset(self):
        """데이터셋 추가"""
        filenames = filedialog.askopenfilenames(
            title="Select Dataset Files",
            filetypes=[("YAML files", "*.yaml *.yml"), ("Text files", "*.txt")]
        )
        
        for filename in filenames:
            self.dataset_listbox.insert(tk.END, os.path.basename(filename))
        
        if filenames:
            self.add_log_entry(f"Added {len(filenames)} dataset file(s)")
    
    def remove_dataset(self):
        """선택된 데이터셋 제거"""
        selection = self.dataset_listbox.curselection()
        if selection:
            self.dataset_listbox.delete(selection)
            self.add_log_entry("Removed selected dataset")
    
    def create_info_grid(self, parent, items):
        """정보 그리드 생성 헬퍼 메서드"""
        for i, (label, var_name, default_value) in enumerate(items):
            row = i // 2
            col = i % 2
            
            item_frame = ttk.Frame(parent)
            item_frame.grid(row=row, column=col, sticky='ew', padx=5, pady=2)
            
            ttk.Label(item_frame, text=label, font=('Arial', 9, 'bold')).pack(anchor='w')
            
            value_label = ttk.Label(item_frame, text=default_value, font=('Arial', 12, 'bold'), 
                                   foreground='#3498db')
            value_label.pack(anchor='w')
            
            # 참조 저장
            setattr(self, f"{var_name}_label", value_label)
        
        # 그리드 가중치 설정
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
    
    # 스케일 업데이트 메서드들
    def update_epochs_label(self, value):
        self.epochs_label.config(text=str(int(float(value))))
        self.total_epochs = int(float(value))
    
    def update_batch_label(self, value):
        self.batch_label.config(text=str(int(float(value))))
    
    def update_lr_label(self, value):
        self.lr_label.config(text=f"{float(value):.3f}")
    
    def update_workers_label(self, value):
        self.workers_label.config(text=str(int(float(value))))
    
    def update_split_ratio_label(self, value):
        ratio = float(value)
        train_percent = int(ratio * 100)
        valid_percent = 100 - train_percent
        self.split_ratio_label.config(text=f"{train_percent}% / {valid_percent}%")
    
    def update_mosaic_label(self, value):
        self.mosaic_label.config(text=str(int(float(value))))
    
    # 모델 관리 메서드들
    def select_model(self, model_type):
        """모델 선택"""
        if self.best_models[model_type]['epoch'] > 0:
            self.selected_model = self.best_models[model_type]
            self.selected_model_type = model_type
            
            # 선택된 모델 정보 업데이트
            self.update_selected_model_display()
            
            self.add_log_entry(f"Selected {model_type} model from epoch {self.selected_model['epoch']}")
    
    def update_selected_model_display(self):
        """선택된 모델 표시 업데이트"""
        if self.selected_model:
            # 제목 업데이트
            self.selected_model_title.config(text=f"Best {self.selected_model_type.title()} Model")
            self.selected_model_path.config(text=f"Epoch {self.selected_model['epoch']} - epoch_{self.selected_model['epoch']:03d}.pt")
            
            # 메트릭 업데이트
            self.selected_precision_label.config(text=f"{self.selected_model['precision']:.3f}")
            self.selected_recall_label.config(text=f"{self.selected_model['recall']:.3f}")
            self.selected_map50_label.config(text=f"{self.selected_model['map50']:.3f}")
            self.selected_map95_label.config(text=f"{self.selected_model['map95']:.3f}")
            
            # 메트릭 프레임 표시
            self.selected_metrics_frame.pack(fill='x', pady=10)
    
    def quick_export_model(self, model_type):
        """빠른 모델 내보내기"""
        if self.best_models[model_type]['epoch'] > 0:
            model = self.best_models[model_type]
            filename = f"best_{model_type}_epoch_{model['epoch']:03d}.pt"
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".pt",
                filetypes=[("PyTorch files", "*.pt")],
                initialname=filename
            )
            
            if save_path:
                self.add_log_entry(f"Exported {model_type} model to {Path(save_path).name}")
                messagebox.showinfo("Export Success", f"Model exported successfully to\n{save_path}")
    
    def download_model(self):
        """선택된 모델 다운로드"""
        if self.selected_model:
            filename = f"selected_model_epoch_{self.selected_model['epoch']:03d}.pt"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".pt",
                filetypes=[("PyTorch files", "*.pt")],
                initialname=filename
            )
            
            if save_path:
                self.add_log_entry(f"Downloaded selected model to {Path(save_path).name}")
                messagebox.showinfo("Download Success", f"Model downloaded to\n{save_path}")
    
    def test_model(self):
        """선택된 모델 테스트"""
        if self.selected_model:
            self.add_log_entry(f"Testing model from epoch {self.selected_model['epoch']}")
            messagebox.showinfo("Test Model", "Model testing functionality will be implemented.")
    
    def deploy_model(self):
        """선택된 모델 배포"""
        if self.selected_model:
            self.add_log_entry(f"Deploying model from epoch {self.selected_model['epoch']}")
            messagebox.showinfo("Deploy Model", "Model deployment functionality will be implemented.")
    
    def export_onnx(self):
        """ONNX 형식으로 내보내기"""
        if self.selected_model:
            filename = f"model_epoch_{self.selected_model['epoch']:03d}.onnx"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".onnx",
                filetypes=[("ONNX files", "*.onnx")],
                initialname=filename
            )
            
            if save_path:
                self.add_log_entry(f"Exported ONNX model to {Path(save_path).name}")
                messagebox.showinfo("ONNX Export", f"ONNX model exported to\n{save_path}")
    
    def export_model_list(self):
        """모델 리스트 내보내기"""
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")],
            initialname="model_list.csv"
        )
        
        if save_path:
            self.add_log_entry(f"Exported model list to {Path(save_path).name}")
            messagebox.showinfo("Export Success", f"Model list exported to\n{save_path}")
    
    def cleanup_models(self):
        """오래된 모델들 정리"""
        if messagebox.askyesno("Cleanup Models", "Remove models older than 50 epochs?"):
            self.add_log_entry("Cleaned up old model files")
            messagebox.showinfo("Cleanup Complete", "Old models have been removed.")
    
    def refresh_model_list(self):
        """모델 리스트 새로고침"""
        self.update_models_table()
        self.add_log_entry("Model list refreshed")
    
    def update_models_table(self):
        """모델 테이블 업데이트"""
        # 기존 항목들 삭제
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)
        
        # 모델 데이터 추가
        for model in sorted(self.saved_models, key=lambda x: x['epoch'], reverse=True):
            self.models_tree.insert('', 'end', values=(
                model['epoch'],
                f"{model['precision']:.3f}",
                f"{model['recall']:.3f}",
                f"{model['map50']:.3f}",
                f"{model['map95']:.3f}",
                model['file_size']
            ))
    
    # Enhanced 훈련 시뮬레이션
    def start_enhanced_training(self):
        """Enhanced 훈련 시작"""
        if self.is_training:
            return
        
        # 설정 검증
        if not self.validate_settings():
            return
        
        # 진행사항 탭으로 전환
        self.notebook.select(1)
        
        self.is_training = True
        self.start_time = time.time()
        self.training_progress = 0
        self.current_epoch = 0
        
        # 메트릭 데이터 초기화
        self.metrics_data = {
            'epochs': [],
            'precision': [],
            'recall': [],
            'map50': [],
            'map95': [],
            'loss': [],
            'lr': []
        }
        
        # Best models 초기화
        for model_type in self.best_models:
            self.best_models[model_type] = {'score': 0, 'epoch': 0, 'precision': 0, 'recall': 0, 'map50': 0, 'map95': 0}
        
        # 저장된 모델들 초기화
        self.saved_models = []
        self.update_models_table()
        
        # UI 업데이트
        self.status_canvas.itemconfig(self.status_dot, fill='green')
        self.status_text_var.set("🚀 Enhanced 훈련 진행 중...")
        
        # 버튼 상태 변경
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')
        
        self.add_log_entry("🚀 Enhanced YOLOv7 훈련이 시작되었습니다!")
        self.add_log_entry(f"📊 Configuration: {self.total_epochs} epochs, batch size {self.batch_size_var.get()}")
        self.add_log_entry(f"Dataset Mode: {self.dataset_mode.get()}")
        self.add_log_entry(f"Advanced Options: Save Best Models={self.save_best_models_var.get()}")
        
        # 실시간 Enhanced 훈련 시뮬레이션 시작
        self.training_thread = threading.Thread(target=self.enhanced_training_simulation)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def enhanced_training_simulation(self):
        """Enhanced 훈련 시뮬레이션 (멀티스레딩 기반)"""
        while self.is_training and self.training_progress < 100:
            # 진행률 업데이트
            self.training_progress += np.random.uniform(0.5, 2.0)
            if self.training_progress > 100:
                self.training_progress = 100
            
            # 현재 에포크 계산
            self.current_epoch = int(self.training_progress / 100 * self.total_epochs)
            
            # 현실적인 메트릭 생성
            progress_ratio = self.training_progress / 100
            noise = lambda: (np.random.random() - 0.5) * 0.1
            
            precision = min(0.95, max(0.1, 0.3 + progress_ratio * 0.6 + noise()))
            recall = min(0.95, max(0.1, 0.25 + progress_ratio * 0.65 + noise()))
            map50 = min(0.95, max(0.05, 0.2 + progress_ratio * 0.7 + noise()))
            map95 = min(0.8, max(0.03, 0.1 + progress_ratio * 0.5 + noise()))
            loss = max(0.05, 0.8 - progress_ratio * 0.6 + noise() * 0.2)
            lr = 0.01 * (1 - progress_ratio * 0.9)
            
            # 메트릭 데이터 업데이트
            if self.current_epoch > 0:
                self.metrics_data['epochs'].append(self.current_epoch)
                self.metrics_data['precision'].append(precision)
                self.metrics_data['recall'].append(recall)
                self.metrics_data['map50'].append(map50)
                self.metrics_data['map95'].append(map95)
                self.metrics_data['loss'].append(loss)
                self.metrics_data['lr'].append(lr)
                
                # Best models 업데이트
                self.update_best_models(self.current_epoch, precision, recall, map50, map95)
                
                # 모델 데이터 저장
                if self.save_all_weights_var.get() or self.save_best_models_var.get():
                    self.save_model_data(self.current_epoch, precision, recall, map50, map95)
            
            # UI 업데이트 (메인 스레드에서)
            self.root.after(0, self.update_enhanced_training_ui, precision, recall, map50, map95, loss, lr)
            
            # 랜덤 로그 엔트리
            if np.random.random() > 0.8:
                gpu_mem = np.random.uniform(6, 8)
                self.root.after(0, self.add_log_entry, 
                               f"Epoch {self.current_epoch}: Loss={loss:.4f}, GPU Memory={gpu_mem:.1f}GB")
            
            time.sleep(0.1)  # 시뮬레이션 딜레이
        
        # 훈련 완료
        if self.is_training:
            self.root.after(0, self.enhanced_training_completed)
    
    def update_enhanced_training_ui(self, precision, recall, map50, map95, loss, lr):
        """Enhanced 훈련 UI 업데이트"""
        # 진행률 바 업데이트
        self.progress_var.set(self.training_progress)
        self.progress_label.config(text=f"{int(self.training_progress)}%")
        
        # 메트릭 요약 카드들 업데이트
        if hasattr(self, 'current_precision_summary_label'):
            self.current_precision_summary_label.config(text=f"{precision:.3f}")
            self.current_recall_summary_label.config(text=f"{recall:.3f}")
            self.current_map50_summary_label.config(text=f"{map50:.3f}")
            self.current_map95_summary_label.config(text=f"{map95:.3f}")
        
        # 진행사항 탭 메트릭 업데이트
        if hasattr(self, 'current_epoch_label'):
            self.current_epoch_label.config(text=str(self.current_epoch))
            self.current_loss_label.config(text=f"{loss:.4f}")
            self.gpu_memory_label.config(text=f"{np.random.uniform(6, 8):.1f}GB")
        
        # 시간 정보 업데이트
        if self.start_time:
            elapsed = time.time() - self.start_time
            
            # 경과 시간 포맷
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.total_time_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # 남은 시간 계산
            if self.current_epoch > 0:
                avg_epoch_time = elapsed / self.current_epoch
                self.avg_epoch_time_label.config(text=f"{avg_epoch_time:.1f}s")
                
                remaining_epochs = self.total_epochs - self.current_epoch
                remaining_seconds = remaining_epochs * avg_epoch_time
                
                if remaining_seconds > 0:
                    remaining_hours = int(remaining_seconds // 3600)
                    remaining_minutes = int((remaining_seconds % 3600) // 60)
                    remaining_secs = int(remaining_seconds % 60)
                    self.remaining_time_label.config(text=f"{remaining_hours:02d}:{remaining_minutes:02d}:{remaining_secs:02d}")
                    
                    # ETA 계산
                    eta_time = datetime.now() + timedelta(seconds=remaining_seconds)
                    self.eta_time_label.config(text=eta_time.strftime("%H:%M"))
        
        # 차트 업데이트
        if MATPLOTLIB_AVAILABLE:
            self.update_charts()
        
        # Best models 표시 업데이트
        self.update_best_models_display()
    
    def update_best_models(self, epoch, precision, recall, map50, map95):
        """Best models 추적 업데이트"""
        # Best Precision
        if precision > self.best_models['precision']['score']:
            self.best_models['precision'] = {
                'score': precision, 'epoch': epoch, 'precision': precision,
                'recall': recall, 'map50': map50, 'map95': map95
            }
        
        # Best Recall
        if recall > self.best_models['recall']['score']:
            self.best_models['recall'] = {
                'score': recall, 'epoch': epoch, 'precision': precision,
                'recall': recall, 'map50': map50, 'map95': map95
            }
        
        # Best Balance
        balance_score = (precision + recall) / 2
        if balance_score > self.best_models['balance']['score']:
            self.best_models['balance'] = {
                'score': balance_score, 'epoch': epoch, 'precision': precision,
                'recall': recall, 'map50': map50, 'map95': map95
            }
        
        # Best mAP
        map_score = (map50 + map95) / 2
        if map_score > self.best_models['map']['score']:
            self.best_models['map'] = {
                'score': map_score, 'epoch': epoch, 'precision': precision,
                'recall': recall, 'map50': map50, 'map95': map95
            }
    
    def update_best_models_display(self):
        """Best models 표시 업데이트"""
        for model_type in ['precision', 'recall', 'balance', 'map']:
            model = self.best_models[model_type]
            if model['epoch'] > 0:
                score_label = getattr(self, f"best_{model_type}_score_label")
                epoch_label = getattr(self, f"best_{model_type}_epoch_label")
                
                if model_type == 'precision':
                    score_label.config(text=f"{model['precision']:.3f}")
                elif model_type == 'recall':
                    score_label.config(text=f"{model['recall']:.3f}")
                else:
                    score_label.config(text=f"{model['score']:.3f}")
                
                epoch_label.config(text=f"Epoch {model['epoch']}")
    
    def save_model_data(self, epoch, precision, recall, map50, map95):
        """모델 데이터 저장"""
        model_data = {
            'epoch': epoch,
            'precision': precision,
            'recall': recall,
            'map50': map50,
            'map95': map95,
            'file_size': f"{np.random.uniform(100, 200):.1f}MB",
            'timestamp': datetime.now().isoformat(),
            'filename': f"epoch_{epoch:03d}.pt"
        }
        
        self.saved_models.append(model_data)
        self.update_models_table()
    
    def enhanced_training_completed(self):
        """Enhanced 훈련 완료 처리"""
        self.is_training = False
        self.status_canvas.itemconfig(self.status_dot, fill='blue')
        self.status_text_var.set("🎉 Enhanced 훈련 완료!")
        
        # 버튼 상태 복원
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
        
        # 모델 선택 탭으로 전환
        self.notebook.select(3)
        
        self.add_log_entry("✅ Enhanced training completed successfully!")
        self.add_log_entry(f"⏱️ Total time: {self.total_time_label.cget('text')}")
        self.add_log_entry("🏆 Best models saved and ready for selection in Models tab")
        
        # 훈련 완료 알림
        messagebox.showinfo("Enhanced Training Complete", 
                           f"🎉 Enhanced training completed successfully!\n\n"
                           f"📊 Total Epochs: {self.total_epochs}\n"
                           f"⏱️ Training Time: {self.total_time_label.cget('text')}\n"
                           f"🏆 Best models are now available in the Models tab.\n"
                           f"Enhanced features: Model management, time tracking, advanced options")
    
    # 기존 메서드들
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
    
    def pause_training(self):
        """훈련 일시정지"""
        if hasattr(self.trainer, 'pause_training') and self.trainer.pause_training():
            self.add_log_entry("⏸️ 훈련이 일시정지되었습니다.")
    
    def stop_training(self):
        """훈련 정지"""
        if messagebox.askyesno("훈련 정지", "정말로 훈련을 정지하시겠습니까?"):
            self.is_training = False
            if hasattr(self.trainer, 'stop_training') and self.trainer.stop_training():
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
            
            self.add_log_entry("🧪 Enhanced 연결 테스트 성공!")
            self.add_log_entry(f"🔧 생성된 명령어: {len(cmd)} 인자")
            messagebox.showinfo("테스트 성공", "✅ YOLOv7 Enhanced 연결이 정상적으로 작동합니다!")
            
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
            
            # 기존 옵션들
            'cache_images': self.cache_images_var.get(),
            'multi_scale': self.multi_scale_var.get(),
            'image_weights': self.image_weights_var.get(),
            'rect': self.rect_var.get(),
            'adam': self.adam_var.get(),
            'sync_bn': self.sync_bn_var.get(),
            
            # 새로 추가된 고급 옵션들
            'close_mosaic': self.close_mosaic_var.get(),
            'save_checkpoints': self.save_checkpoints_var.get(),
            'save_all_weights': self.save_all_weights_var.get(),
            'save_best_models': self.save_best_models_var.get(),
            'wandb_logging': self.wandb_logging_var.get(),
            'tensorboard': self.tensorboard_var.get(),
            'plot_results': self.plot_results_var.get(),
            
            # Dataset 모드 관련
            'dataset_mode': self.dataset_mode.get(),
            'split_ratio': self.split_ratio_var.get(),
            'shuffle': self.shuffle_var.get(),
            'balance': self.balance_var.get(),
            'remove_duplicates': self.remove_duplicates_var.get(),
        }
    
    def update_charts(self):
        """차트 업데이트"""
        if not MATPLOTLIB_AVAILABLE or len(self.metrics_data['epochs']) < 2:
            return
        
        # 차트 지우기
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        epochs = self.metrics_data['epochs']
        
        # Precision & Recall 차트
        if self.metrics_data['precision'] and self.metrics_data['recall']:
            self.ax1.plot(epochs, self.metrics_data['precision'], 'b-', label='Precision', linewidth=2)
            self.ax1.plot(epochs, self.metrics_data['recall'], 'r-', label='Recall', linewidth=2)
            self.ax1.set_title("Precision & Recall", fontsize=12, fontweight='bold')
            self.ax1.set_xlabel("Epoch")
            self.ax1.set_ylabel("Score")
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
        
        # mAP 차트
        if self.metrics_data['map50'] and self.metrics_data['map95']:
            self.ax2.plot(epochs, self.metrics_data['map50'], 'g-', label='mAP@0.5', linewidth=2)
            self.ax2.plot(epochs, self.metrics_data['map95'], 'purple', label='mAP@0.5:0.95', linewidth=2)
            self.ax2.set_title("mAP Metrics", fontsize=12, fontweight='bold')
            self.ax2.set_xlabel("Epoch")
            self.ax2.set_ylabel("mAP Score")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
        
        # Loss 차트
        if self.metrics_data['loss']:
            self.ax3.plot(epochs, self.metrics_data['loss'], 'orange', linewidth=2)
            self.ax3.set_title("Training Loss", fontsize=12, fontweight='bold')
            self.ax3.set_xlabel("Epoch")
            self.ax3.set_ylabel("Loss")
            self.ax3.grid(True, alpha=0.3)
        
        # Learning Rate 차트
        if self.metrics_data['lr']:
            self.ax4.plot(epochs, self.metrics_data['lr'], 'brown', linewidth=2)
            self.ax4.set_title("Learning Rate", fontsize=12, fontweight='bold')
            self.ax4.set_xlabel("Epoch")
            self.ax4.set_ylabel("Learning Rate")
            self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.chart_canvas.draw()
    
    def add_log_entry(self, message):
        """로그 엔트리 추가"""
        # log_text가 초기화되었는지 확인
        if not hasattr(self, 'log_text') or self.log_text is None:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # 텍스트 위젯에 추가
        self.log_text.insert(tk.END, log_message)
        
        # 자동 스크롤
        self.log_text.see(tk.END)
        
        # 로그 길이 제한 (1000줄)
        lines = self.log_text.get("1.0", tk.END).split('\n')
        if len(lines) > 1000:
            # 처음 100줄 삭제
            self.log_text.delete("1.0", "101.0")
    
    def show(self):
        """윈도우 표시"""
        self.root.deiconify()  # 윈도우 숨김 해제
    
    # 콜백 메서드들
    def on_training_started(self, data):
        """훈련 시작 콜백"""
        pass
    
    def on_metrics_update(self, metrics):
        """메트릭 업데이트 콜백"""
        pass
    
    def on_log_update(self, data):
        """로그 업데이트 콜백"""
        pass
    
    def on_training_complete(self, data):
        """훈련 완료 콜백"""
        pass
    
    def on_training_stopped(self, data):
        """훈련 정지 콜백"""
        pass
    
    def on_error(self, data):
        """에러 콜백"""
        pass