"""
데이터셋 병합 모듈
여러 데이터셋을 하나로 통합하는 기능
"""

import sys
import io
import os
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
import platform

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == 'win32':
    try:
        if sys.version_info >= (3, 7):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class DatasetMerger:
    """여러 YOLO 데이터셋을 하나로 병합하는 클래스"""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: 병합된 데이터셋을 저장할 디렉토리
        """
        self.output_dir = output_dir or Path("merged_dataset")
        self.datasets = []
        self.merged_classes = OrderedDict()
        self.class_mapping = {}  # 각 데이터셋의 클래스 인덱스 매핑

    def add_dataset(self, dataset_path: Path, data_yaml_path: Optional[Path] = None):
        """
        데이터셋 추가

        Args:
            dataset_path: 데이터셋 경로
            data_yaml_path: data.yaml 파일 경로 (선택)
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"데이터셋 경로가 존재하지 않습니다: {dataset_path}")

        # data.yaml 찾기
        if data_yaml_path is None:
            data_yaml_path = dataset_path / "data.yaml"
            if not data_yaml_path.exists():
                data_yaml_path = dataset_path / "dataset.yaml"

        # data.yaml 로드
        dataset_info = self._load_dataset_info(dataset_path, data_yaml_path)

        # 데이터셋 추가
        self.datasets.append(dataset_info)

        # 클래스 병합
        self._merge_classes(dataset_info)

        return dataset_info

    def _load_dataset_info(self, dataset_path: Path, data_yaml_path: Path) -> Dict[str, Any]:
        """데이터셋 정보 로드"""
        info = {
            'path': dataset_path,
            'data_yaml': data_yaml_path,
            'classes': [],
            'nc': 0,
            'train_images': [],
            'valid_images': []
        }

        # data.yaml 읽기
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)

            info['classes'] = data_config.get('names', [])
            info['nc'] = data_config.get('nc', len(info['classes']))
            info['original_config'] = data_config

        # train/valid 이미지 찾기
        info['train_images'] = self._find_images(dataset_path, 'train')
        info['valid_images'] = self._find_images(dataset_path, 'valid')

        return info

    def _find_images(self, dataset_path: Path, split: str) -> List[Path]:
        """이미지 파일들 찾기"""
        images = []

        # 일반적인 YOLO 구조 탐색
        search_paths = [
            dataset_path / 'images' / split,
            dataset_path / split / 'images',
            dataset_path / split,
            dataset_path / 'train' if split == 'train' else dataset_path / 'valid',
            dataset_path / 'val' if split == 'valid' else None
        ]

        search_paths = [p for p in search_paths if p is not None]

        for search_path in search_paths:
            if search_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images.extend(list(search_path.glob(ext)))
                    images.extend(list(search_path.glob(ext.upper())))

        return images

    def _merge_classes(self, dataset_info: Dict[str, Any]):
        """클래스 정보 병합"""
        dataset_classes = dataset_info['classes']
        dataset_idx = len(self.datasets) - 1

        # 각 데이터셋의 클래스를 전역 클래스 리스트에 매핑
        class_map = {}

        for local_idx, class_name in enumerate(dataset_classes):
            if class_name not in self.merged_classes:
                # 새로운 클래스 추가
                global_idx = len(self.merged_classes)
                self.merged_classes[class_name] = global_idx
            else:
                # 기존 클래스 사용
                global_idx = self.merged_classes[class_name]

            class_map[local_idx] = global_idx

        self.class_mapping[dataset_idx] = class_map

    def merge(self, method: str = 'symlink', show_progress=None) -> Dict[str, Any]:
        """
        데이터셋 병합

        Args:
            method: 병합 방식 ('symlink', 'list', 'copy')
            show_progress: 진행률 표시 콜백 함수

        Returns:
            병합 결과 정보
        """
        if not self.datasets:
            raise ValueError("병합할 데이터셋이 없습니다. add_dataset()으로 먼저 추가하세요.")

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 병합 방식에 따라 처리
        if method == 'symlink':
            result = self._merge_symlink(show_progress)
        elif method == 'list':
            result = self._merge_list(show_progress)
        elif method == 'copy':
            result = self._merge_copy(show_progress)
        else:
            raise ValueError(f"지원하지 않는 병합 방식: {method}")

        # data.yaml 생성
        self._create_merged_yaml()

        return result

    def _merge_symlink(self, show_progress=None) -> Dict[str, Any]:
        """심볼릭 링크 방식으로 병합"""
        is_windows = platform.system() == 'Windows'

        if is_windows:
            print("⚠️ Windows에서는 symlink가 제한적입니다. 관리자 권한이 필요할 수 있습니다.")

        # 디렉토리 구조 생성
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'valid').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'valid').mkdir(parents=True, exist_ok=True)

        total_train = sum(len(ds['train_images']) for ds in self.datasets)
        total_valid = sum(len(ds['valid_images']) for ds in self.datasets)
        processed = 0
        total = total_train + total_valid

        for dataset_idx, dataset_info in enumerate(self.datasets):
            # Train 이미지 처리
            for img_path in dataset_info['train_images']:
                self._create_symlink_pair(
                    img_path,
                    self.output_dir / 'images' / 'train',
                    self.output_dir / 'labels' / 'train',
                    dataset_idx
                )
                processed += 1
                if show_progress and total > 0:
                    show_progress(processed / total * 100)

            # Valid 이미지 처리
            for img_path in dataset_info['valid_images']:
                self._create_symlink_pair(
                    img_path,
                    self.output_dir / 'images' / 'valid',
                    self.output_dir / 'labels' / 'valid',
                    dataset_idx
                )
                processed += 1
                if show_progress and total > 0:
                    show_progress(processed / total * 100)

        return {
            'method': 'symlink',
            'train_count': total_train,
            'valid_count': total_valid,
            'total': total
        }

    def _merge_list(self, show_progress=None) -> Dict[str, Any]:
        """리스트 파일 방식으로 병합 (train.txt, valid.txt)"""
        train_list = []
        valid_list = []

        # 디렉토리 구조 생성 (라벨만)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'valid').mkdir(parents=True, exist_ok=True)

        total_train = sum(len(ds['train_images']) for ds in self.datasets)
        total_valid = sum(len(ds['valid_images']) for ds in self.datasets)
        processed = 0
        total = total_train + total_valid

        for dataset_idx, dataset_info in enumerate(self.datasets):
            # Train 이미지 처리
            for img_path in dataset_info['train_images']:
                train_list.append(str(img_path.absolute()))
                # 라벨 파일 복사 (클래스 인덱스 변환)
                self._copy_label_with_remap(
                    img_path,
                    self.output_dir / 'labels' / 'train',
                    dataset_idx
                )
                processed += 1
                if show_progress and total > 0:
                    show_progress(processed / total * 100)

            # Valid 이미지 처리
            for img_path in dataset_info['valid_images']:
                valid_list.append(str(img_path.absolute()))
                # 라벨 파일 복사 (클래스 인덱스 변환)
                self._copy_label_with_remap(
                    img_path,
                    self.output_dir / 'labels' / 'valid',
                    dataset_idx
                )
                processed += 1
                if show_progress and total > 0:
                    show_progress(processed / total * 100)

        # train.txt, valid.txt 생성
        with open(self.output_dir / 'train.txt', 'w') as f:
            f.write('\n'.join(train_list))

        with open(self.output_dir / 'valid.txt', 'w') as f:
            f.write('\n'.join(valid_list))

        return {
            'method': 'list',
            'train_count': len(train_list),
            'valid_count': len(valid_list),
            'total': len(train_list) + len(valid_list)
        }

    def _merge_copy(self, show_progress=None) -> Dict[str, Any]:
        """파일 복사 방식으로 병합"""
        # 디렉토리 구조 생성
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'valid').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'valid').mkdir(parents=True, exist_ok=True)

        total_train = sum(len(ds['train_images']) for ds in self.datasets)
        total_valid = sum(len(ds['valid_images']) for ds in self.datasets)
        processed = 0
        total = total_train + total_valid

        for dataset_idx, dataset_info in enumerate(self.datasets):
            # Train 이미지 처리
            for img_path in dataset_info['train_images']:
                self._copy_image_label_pair(
                    img_path,
                    self.output_dir / 'images' / 'train',
                    self.output_dir / 'labels' / 'train',
                    dataset_idx
                )
                processed += 1
                if show_progress and total > 0:
                    show_progress(processed / total * 100)

            # Valid 이미지 처리
            for img_path in dataset_info['valid_images']:
                self._copy_image_label_pair(
                    img_path,
                    self.output_dir / 'images' / 'valid',
                    self.output_dir / 'labels' / 'valid',
                    dataset_idx
                )
                processed += 1
                if show_progress and total > 0:
                    show_progress(processed / total * 100)

        return {
            'method': 'copy',
            'train_count': total_train,
            'valid_count': total_valid,
            'total': total
        }

    def _create_symlink_pair(self, img_path: Path, img_dest_dir: Path,
                            label_dest_dir: Path, dataset_idx: int):
        """이미지와 라벨 심볼릭 링크 생성"""
        # 고유한 파일명 생성 (충돌 방지)
        unique_name = f"ds{dataset_idx}_{img_path.name}"

        # 이미지 symlink
        img_link = img_dest_dir / unique_name
        if not img_link.exists():
            try:
                img_link.symlink_to(img_path.absolute())
            except OSError as e:
                # Windows에서 권한 문제 시 복사로 대체
                shutil.copy2(img_path, img_link)

        # 라벨 파일 처리 (클래스 인덱스 변환 필요)
        self._copy_label_with_remap(img_path, label_dest_dir, dataset_idx, unique_name)

    def _copy_label_with_remap(self, img_path: Path, label_dest_dir: Path,
                               dataset_idx: int, custom_name: Optional[str] = None):
        """라벨 파일 복사하면서 클래스 인덱스 재매핑"""
        # 라벨 파일 경로 찾기
        label_path = self._find_label_path(img_path)

        if label_path is None or not label_path.exists():
            return

        # 라벨 파일 읽기
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 클래스 인덱스 재매핑
        class_map = self.class_mapping[dataset_idx]
        remapped_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # class x y w h
                old_class_idx = int(parts[0])
                new_class_idx = class_map.get(old_class_idx, old_class_idx)
                remapped_lines.append(f"{new_class_idx} {' '.join(parts[1:])}\n")

        # 새 라벨 파일 저장
        if custom_name:
            label_name = Path(custom_name).stem + '.txt'
        else:
            label_name = f"ds{dataset_idx}_{img_path.stem}.txt"

        label_dest = label_dest_dir / label_name

        with open(label_dest, 'w') as f:
            f.writelines(remapped_lines)

    def _copy_image_label_pair(self, img_path: Path, img_dest_dir: Path,
                               label_dest_dir: Path, dataset_idx: int):
        """이미지와 라벨 파일 복사"""
        # 고유한 파일명 생성
        unique_name = f"ds{dataset_idx}_{img_path.name}"

        # 이미지 복사
        img_dest = img_dest_dir / unique_name
        shutil.copy2(img_path, img_dest)

        # 라벨 복사 (클래스 인덱스 변환)
        self._copy_label_with_remap(img_path, label_dest_dir, dataset_idx, unique_name)

    def _find_label_path(self, img_path: Path) -> Optional[Path]:
        """이미지 경로로부터 라벨 파일 경로 찾기"""
        # 라벨 경로 후보들
        possible_label_paths = [
            # images/train/xxx.jpg -> labels/train/xxx.txt
            Path(str(img_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')).with_suffix('.txt'),
            # train/xxx.jpg -> labels/train/xxx.txt
            img_path.parent.parent / 'labels' / img_path.parent.name / (img_path.stem + '.txt'),
            # train/images/xxx.jpg -> train/labels/xxx.txt
            img_path.parent.parent / 'labels' / (img_path.stem + '.txt'),
            # xxx.jpg -> xxx.txt (같은 폴더)
            img_path.with_suffix('.txt')
        ]

        for label_path in possible_label_paths:
            if label_path.exists():
                return label_path

        return None

    def _create_merged_yaml(self):
        """병합된 data.yaml 생성"""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train' if (self.output_dir / 'images' / 'train').exists() else 'train.txt',
            'val': 'images/valid' if (self.output_dir / 'images' / 'valid').exists() else 'valid.txt',
            'nc': len(self.merged_classes),
            'names': list(self.merged_classes.keys())
        }

        yaml_path = self.output_dir / 'data.yaml'

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"✅ data.yaml 생성 완료: {yaml_path}")
        return yaml_path

    def get_merge_summary(self) -> str:
        """병합 요약 정보 반환"""
        summary = []
        summary.append("=" * 50)
        summary.append("📊 데이터셋 병합 요약")
        summary.append("=" * 50)
        summary.append(f"총 데이터셋 개수: {len(self.datasets)}")
        summary.append(f"병합된 클래스 개수: {len(self.merged_classes)}")
        summary.append(f"클래스 목록: {', '.join(self.merged_classes.keys())}")
        summary.append("")

        total_train = sum(len(ds['train_images']) for ds in self.datasets)
        total_valid = sum(len(ds['valid_images']) for ds in self.datasets)

        summary.append(f"총 Train 이미지: {total_train}")
        summary.append(f"총 Valid 이미지: {total_valid}")
        summary.append(f"총 이미지: {total_train + total_valid}")
        summary.append("")

        summary.append("데이터셋별 상세:")
        for i, ds in enumerate(self.datasets):
            summary.append(f"  Dataset {i+1}: {ds['path'].name}")
            summary.append(f"    - Train: {len(ds['train_images'])}")
            summary.append(f"    - Valid: {len(ds['valid_images'])}")
            summary.append(f"    - Classes: {ds['nc']} ({', '.join(ds['classes'])})")

        summary.append("=" * 50)

        return '\n'.join(summary)


# 테스트 코드
if __name__ == "__main__":
    print("🧪 DatasetMerger 테스트...")

    merger = DatasetMerger(Path("test_merged_dataset"))

    # 테스트용으로 가상 데이터셋 정보 생성
    print("✅ DatasetMerger 초기화 성공!")
    print(f"출력 디렉토리: {merger.output_dir}")
