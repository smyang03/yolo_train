"""
Utils module - 공통 유틸리티 함수들
"""

def safe_print(*args, **kwargs):
    """
    안전한 print 함수 - PyInstaller EXE에서 stdout이 닫혀있을 때도 동작

    PyInstaller로 빌드된 EXE에서는 stdout/stderr이 닫혀있거나
    존재하지 않을 수 있습니다. 이 함수는 그런 경우에도 에러를 발생시키지 않습니다.
    """
    try:
        print(*args, **kwargs)
    except (ValueError, OSError, AttributeError):
        # stdout/stderr이 닫혀있거나 없는 경우 무시
        pass

# 편의를 위해 다른 유틸리티 함수들도 export
__all__ = ['safe_print']
