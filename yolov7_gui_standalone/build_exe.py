"""
build_exe.py
PyInstaller를 사용한 실행 파일 빌드 스크립트
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


class ExeBuilder:
    """EXE 빌드 관리 클래스"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.spec_file = self.project_root / "yolov7_gui.spec"

        # 빌드에 포함할 데이터 파일들 (경로가 존재하는 것만 추가)
        self.datas = []

        # resources 디렉토리 확인 및 추가
        resources_path = self.project_root / "resources"
        if resources_path.exists():
            self.datas.append((str(resources_path), "resources"))
        else:
            print(f"⚠️ 경고: resources 디렉토리를 찾을 수 없습니다: {resources_path}")

        # yolov7_embedded 디렉토리 확인 및 추가
        yolov7_embedded_path = self.project_root / "yolov7_embedded"
        if yolov7_embedded_path.exists():
            self.datas.append((str(yolov7_embedded_path), "yolov7_embedded"))
        else:
            print(f"⚠️ 경고: yolov7_embedded 디렉토리를 찾을 수 없습니다: {yolov7_embedded_path}")

        # 숨겨진 import들 (PyInstaller가 자동 감지 못하는 모듈)
        self.hidden_imports = [
            # 딥러닝 프레임워크
            'torch',
            'torchvision',
            'torch.nn',
            'torch.optim',
            'torch.utils',
            'torch.utils.data',

            # 컴퓨터 비전
            'cv2',
            'PIL',
            'PIL.Image',
            'albumentations',

            # 수치 연산
            'numpy',
            'pandas',
            'scipy',
            'sklearn',

            # GUI 및 시각화
            'matplotlib',
            'matplotlib.pyplot',
            'matplotlib.backends.backend_tkagg',
            'matplotlib.figure',

            # 설정 및 유틸리티
            'yaml',
            'json',
            'pathlib',
            'tqdm',
            'queue',
            'threading',
            'subprocess',

            # YOLO 관련
            'yolov7_embedded',
            'yolov7_embedded.train_core',
        ]

    def clean_build(self):
        """빌드 디렉토리 정리"""
        print("🧹 이전 빌드 파일 정리 중...")

        dirs_to_clean = [self.dist_dir, self.build_dir]
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   삭제: {dir_path}")

        if self.spec_file.exists():
            self.spec_file.unlink()
            print(f"   삭제: {self.spec_file}")

        print("✅ 정리 완료\n")

    def create_spec_file(self):
        """PyInstaller spec 파일 생성"""
        print("📝 Spec 파일 생성 중...")

        # datas 문자열 생성 (경로를 슬래시로 변환하여 Windows/Linux 호환성 확보)
        datas_str = ", ".join([f"(r'{d[0]}', '{d[1]}')" for d in self.datas])

        # hidden imports 문자열 생성
        hidden_imports_str = ", ".join([f"'{m}'" for m in self.hidden_imports])

        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[r'{str(self.project_root)}'],
    binaries=[],
    datas=[{datas_str}],
    hiddenimports=[{hidden_imports_str}],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter.test', 'test', 'unittest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YOLOv7_Training_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI 애플리케이션이므로 콘솔 창 숨김
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 아이콘 파일이 있으면 경로 지정
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YOLOv7_Training_GUI',
)
"""

        with open(self.spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)

        print(f"✅ Spec 파일 생성 완료: {self.spec_file}\n")

    def build_onefile(self):
        """단일 EXE 파일 빌드 (모든 것을 하나의 파일로 패키징)"""
        print("🔨 단일 EXE 파일 빌드 시작...")
        print("⚠️ 경고: 단일 파일 모드는 시작 시간이 느릴 수 있습니다.\n")

        # 플랫폼별 경로 구분자 결정 (Windows: ;, Linux/Mac: :)
        separator = ';' if os.name == 'nt' else ':'

        # 기본 PyInstaller 명령어
        cmd = [
            'pyinstaller',
            '--name=YOLOv7_Training_GUI',
            '--onefile',  # 단일 파일로 빌드
            '--windowed',  # GUI 모드 (콘솔 숨김)
            '--clean',
        ]

        # 데이터 파일 추가
        for data_src, data_dst in self.datas:
            cmd.append(f'--add-data={data_src}{separator}{data_dst}')

        # 숨겨진 import 추가
        for module in self.hidden_imports:
            cmd.append(f'--hidden-import={module}')

        # 메인 스크립트
        cmd.append('main.py')

        self._run_build(cmd)

    def build_onedir(self):
        """디렉토리 형태로 빌드 (권장)"""
        print("🔨 디렉토리 형태 EXE 빌드 시작...")
        print("✅ 권장: 시작 속도가 빠르고 디버깅이 쉽습니다.\n")

        # spec 파일 사용
        if not self.spec_file.exists():
            self.create_spec_file()

        cmd = [
            'pyinstaller',
            '--clean',
            str(self.spec_file)
        ]

        self._run_build(cmd)

    def _run_build(self, cmd):
        """빌드 명령 실행"""
        try:
            print(f"실행 명령: {' '.join(cmd)}\n")
            print("=" * 70)

            # 빌드 실행
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(self.project_root)
            )

            # 실시간 출력
            for line in process.stdout:
                print(line, end='')

            process.wait()

            print("=" * 70)

            if process.returncode == 0:
                print("\n✅ 빌드 성공!")
                self._show_build_info()
            else:
                print(f"\n❌ 빌드 실패 (코드: {process.returncode})")
                sys.exit(1)

        except FileNotFoundError:
            print("❌ PyInstaller를 찾을 수 없습니다.")
            print("설치 명령: pip install pyinstaller")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 빌드 오류: {e}")
            sys.exit(1)

    def _show_build_info(self):
        """빌드 결과 정보 표시"""
        print("\n" + "=" * 70)
        print("📦 빌드 결과")
        print("=" * 70)

        if self.dist_dir.exists():
            print(f"출력 디렉토리: {self.dist_dir}")

            # 디렉토리 내용 표시
            for item in self.dist_dir.rglob('*'):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"  📄 {item.name} ({size_mb:.2f} MB)")

            # 전체 크기 계산
            total_size = sum(f.stat().st_size for f in self.dist_dir.rglob('*') if f.is_file())
            print(f"\n전체 크기: {total_size / (1024 * 1024):.2f} MB")

        print("=" * 70)
        print("\n⚠️  중요: EXE 실행 전 확인사항")
        print("=" * 70)
        print("1. YOLOv7 레포지토리가 필요합니다:")
        print("   - dist/ 폴더와 같은 위치에 yolov7/ 폴더 배치")
        print("   - 또는 환경 변수 설정: set YOLOV7_PATH=C:\\path\\to\\yolov7")
        print("")
        print("2. CUDA 및 cuDNN이 설치되어 있어야 GPU 사용 가능")
        print("")
        print("3. 첫 실행 시 시간이 걸릴 수 있습니다")
        print("=" * 70)

    def create_installer_script(self):
        """Inno Setup 인스톨러 스크립트 생성 (옵션)"""
        print("📝 인스톨러 스크립트 생성 중...")

        inno_script = """
[Setup]
AppName=YOLOv7 Training GUI
AppVersion=1.0.0
DefaultDirName={autopf}\\YOLOv7_Training_GUI
DefaultGroupName=YOLOv7 Training GUI
OutputDir=installer
OutputBaseFilename=YOLOv7_Training_GUI_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\\YOLOv7_Training_GUI\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\YOLOv7 Training GUI"; Filename: "{app}\\YOLOv7_Training_GUI.exe"
Name: "{autodesktop}\\YOLOv7 Training GUI"; Filename: "{app}\\YOLOv7_Training_GUI.exe"

[Run]
Filename: "{app}\\YOLOv7_Training_GUI.exe"; Description: "Launch YOLOv7 Training GUI"; Flags: nowait postinstall skipifsilent
"""

        installer_script_path = self.project_root / "installer_script.iss"
        with open(installer_script_path, 'w', encoding='utf-8') as f:
            f.write(inno_script.strip())

        print(f"✅ 인스톨러 스크립트 생성 완료: {installer_script_path}")
        print("   Inno Setup으로 컴파일하여 설치 파일을 만들 수 있습니다.")
        print("   다운로드: https://jrsoftware.org/isdl.php\n")

    def test_exe(self):
        """빌드된 EXE 파일 테스트 실행"""
        print("🧪 EXE 파일 테스트 실행...")

        exe_path = self.dist_dir / "YOLOv7_Training_GUI" / "YOLOv7_Training_GUI.exe"

        if not exe_path.exists():
            # 단일 파일 모드 경로 확인
            exe_path = self.dist_dir / "YOLOv7_Training_GUI.exe"

        if exe_path.exists():
            print(f"실행: {exe_path}")
            subprocess.Popen([str(exe_path)])
        else:
            print(f"❌ EXE 파일을 찾을 수 없습니다: {exe_path}")


def main():
    """메인 함수"""
    print("=" * 70)
    print("🚀 YOLOv7 Training GUI - EXE 빌드 스크립트")
    print("=" * 70)
    print()

    builder = ExeBuilder()

    # 명령줄 인수 처리
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'clean':
            builder.clean_build()
            return

        elif command == 'onefile':
            builder.clean_build()
            builder.build_onefile()
            return

        elif command == 'onedir':
            builder.clean_build()
            builder.build_onedir()
            return

        elif command == 'test':
            builder.test_exe()
            return

        elif command == 'installer':
            builder.create_installer_script()
            return

        elif command == 'all':
            builder.clean_build()
            builder.build_onedir()
            builder.create_installer_script()
            print("\n✅ 모든 빌드 작업 완료!")
            return

        else:
            print(f"❌ 알 수 없는 명령: {command}")
            print_usage()
            return

    # 기본 동작: 메뉴 표시
    print("빌드 옵션을 선택하세요:")
    print("1. 디렉토리 형태 빌드 (권장)")
    print("2. 단일 파일 빌드")
    print("3. 이전 빌드 정리")
    print("4. 전체 빌드 (정리 + 빌드 + 인스톨러)")
    print("5. EXE 테스트 실행")
    print("0. 종료")
    print()

    try:
        choice = input("선택 (0-5): ").strip()

        if choice == '1':
            builder.clean_build()
            builder.build_onedir()

        elif choice == '2':
            builder.clean_build()
            builder.build_onefile()

        elif choice == '3':
            builder.clean_build()

        elif choice == '4':
            builder.clean_build()
            builder.build_onedir()
            builder.create_installer_script()
            print("\n✅ 모든 빌드 작업 완료!")

        elif choice == '5':
            builder.test_exe()

        elif choice == '0':
            print("종료합니다.")

        else:
            print("❌ 잘못된 선택입니다.")
            print_usage()

    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


def print_usage():
    """사용법 출력"""
    print("\n사용법:")
    print("  python build_exe.py              # 대화형 메뉴")
    print("  python build_exe.py onedir       # 디렉토리 형태 빌드 (권장)")
    print("  python build_exe.py onefile      # 단일 파일 빌드")
    print("  python build_exe.py clean        # 빌드 파일 정리")
    print("  python build_exe.py installer    # 인스톨러 스크립트 생성")
    print("  python build_exe.py all          # 전체 빌드")
    print("  python build_exe.py test         # EXE 테스트 실행")
    print()


if __name__ == "__main__":
    main()
