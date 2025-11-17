"""
src/utils/file_utils.py
파일 유틸리티 모듈
"""

import os
import shutil
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def ensure_dir(directory: Path) -> Path:
    """디렉토리가 존재하는지 확인하고 없으면 생성

    Args:
        directory: 생성할 디렉토리 경로

    Returns:
        Path: 생성된 디렉토리 경로
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """파일 복사

    Args:
        src: 원본 파일 경로
        dst: 대상 파일 경로
        overwrite: 덮어쓰기 여부

    Returns:
        bool: 성공 여부
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            safe_print(f"원본 파일을 찾을 수 없습니다: {src}")
            return False

        if dst_path.exists() and not overwrite:
            safe_print(f"대상 파일이 이미 존재합니다: {dst}")
            return False

        # 대상 디렉토리 생성
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 복사
        shutil.copy2(src_path, dst_path)
        return True

    except Exception as e:
        safe_print(f"파일 복사 실패: {e}")
        return False


def move_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """파일 이동

    Args:
        src: 원본 파일 경로
        dst: 대상 파일 경로
        overwrite: 덮어쓰기 여부

    Returns:
        bool: 성공 여부
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            safe_print(f"원본 파일을 찾을 수 없습니다: {src}")
            return False

        if dst_path.exists() and not overwrite:
            safe_print(f"대상 파일이 이미 존재합니다: {dst}")
            return False

        # 대상 디렉토리 생성
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 이동
        shutil.move(str(src_path), str(dst_path))
        return True

    except Exception as e:
        safe_print(f"파일 이동 실패: {e}")
        return False


def delete_file(file_path: str) -> bool:
    """파일 삭제

    Args:
        file_path: 삭제할 파일 경로

    Returns:
        bool: 성공 여부
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False
    except Exception as e:
        safe_print(f"파일 삭제 실패: {e}")
        return False


def get_file_size(file_path: str, unit: str = 'MB') -> float:
    """파일 크기 반환

    Args:
        file_path: 파일 경로
        unit: 단위 ('B', 'KB', 'MB', 'GB')

    Returns:
        float: 파일 크기
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return 0.0

        size_bytes = path.stat().st_size

        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        divisor = units.get(unit.upper(), 1024**2)

        return size_bytes / divisor

    except Exception as e:
        safe_print(f"파일 크기 확인 실패: {e}")
        return 0.0


def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[Path]:
    """디렉토리 내 파일 목록 반환

    Args:
        directory: 검색할 디렉토리
        pattern: 파일 패턴 (예: "*.pt", "*.yaml")
        recursive: 하위 디렉토리 포함 여부

    Returns:
        List[Path]: 파일 경로 리스트
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # 파일만 필터링
        return [f for f in files if f.is_file()]

    except Exception as e:
        safe_print(f"파일 목록 조회 실패: {e}")
        return []


def read_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """YAML 파일 읽기

    Args:
        file_path: YAML 파일 경로

    Returns:
        Optional[Dict]: 파싱된 데이터 또는 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        safe_print(f"YAML 파일 읽기 실패: {e}")
        return None


def write_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """YAML 파일 쓰기

    Args:
        data: 저장할 데이터
        file_path: 저장할 파일 경로

    Returns:
        bool: 성공 여부
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        safe_print(f"YAML 파일 쓰기 실패: {e}")
        return False


def read_json(file_path: str) -> Optional[Dict[str, Any]]:
    """JSON 파일 읽기

    Args:
        file_path: JSON 파일 경로

    Returns:
        Optional[Dict]: 파싱된 데이터 또는 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        safe_print(f"JSON 파일 읽기 실패: {e}")
        return None


def write_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> bool:
    """JSON 파일 쓰기

    Args:
        data: 저장할 데이터
        file_path: 저장할 파일 경로
        indent: 들여쓰기 수준

    Returns:
        bool: 성공 여부
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        safe_print(f"JSON 파일 쓰기 실패: {e}")
        return False


def get_timestamp_filename(prefix: str = "", extension: str = ".txt") -> str:
    """타임스탬프를 포함한 파일명 생성

    Args:
        prefix: 파일명 접두사
        extension: 파일 확장자

    Returns:
        str: 타임스탬프 파일명
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        return f"{prefix}_{timestamp}{extension}"
    return f"{timestamp}{extension}"


def cleanup_old_files(directory: str, days: int = 7, pattern: str = "*") -> int:
    """오래된 파일 정리

    Args:
        directory: 정리할 디렉토리
        days: 보관 기간 (일)
        pattern: 파일 패턴

    Returns:
        int: 삭제된 파일 수
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (days * 24 * 3600)

        deleted_count = 0
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_time = file_path.stat().st_mtime
                if file_time < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1

        return deleted_count

    except Exception as e:
        safe_print(f"파일 정리 실패: {e}")
        return 0


def get_directory_size(directory: str, unit: str = 'MB') -> float:
    """디렉토리 전체 크기 계산

    Args:
        directory: 디렉토리 경로
        unit: 단위 ('B', 'KB', 'MB', 'GB')

    Returns:
        float: 디렉토리 크기
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0.0

        total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())

        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        divisor = units.get(unit.upper(), 1024**2)

        return total_size / divisor

    except Exception as e:
        safe_print(f"디렉토리 크기 계산 실패: {e}")
        return 0.0


def create_backup(file_path: str, backup_dir: Optional[str] = None) -> bool:
    """파일 백업

    Args:
        file_path: 백업할 파일 경로
        backup_dir: 백업 디렉토리 (None이면 같은 디렉토리에 .bak 생성)

    Returns:
        bool: 성공 여부
    """
    try:
        src_path = Path(file_path)
        if not src_path.exists():
            return False

        if backup_dir:
            backup_path = Path(backup_dir) / f"{src_path.name}.bak"
            ensure_dir(backup_path.parent)
        else:
            backup_path = src_path.with_suffix(src_path.suffix + '.bak')

        shutil.copy2(src_path, backup_path)
        return True

    except Exception as e:
        safe_print(f"백업 생성 실패: {e}")
        return False


# 사용 예시
if __name__ == "__main__":
    # 테스트
    safe_print("파일 유틸리티 테스트")

    # 디렉토리 생성
    test_dir = ensure_dir("./test_output")
    safe_print(f"디렉토리 생성: {test_dir}")

    # YAML 쓰기/읽기 테스트
    test_data = {
        'name': 'test',
        'value': 123,
        'list': [1, 2, 3]
    }

    yaml_file = test_dir / "test.yaml"
    if write_yaml(test_data, str(yaml_file)):
        safe_print(f"YAML 파일 저장: {yaml_file}")
        loaded = read_yaml(str(yaml_file))
        safe_print(f"YAML 파일 로드: {loaded}")

    # 파일 크기 확인
    size = get_file_size(str(yaml_file))
    safe_print(f"파일 크기: {size:.2f} MB")

    # 파일 목록
    files = list_files(str(test_dir), "*.yaml")
    safe_print(f"YAML 파일들: {[f.name for f in files]}")
