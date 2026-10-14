# RusTuberV

[日本語](README.md) · [English](README.en.md) · [简体中文](README.zh-CN.md) · **한국어**

RusTuberV는 Rust와 Bevy로 구현한 웹캠 기반 VRM 아바타 애플리케이션입니다. MediaPipe의 얼굴·자세·손 추론 결과를 아바타에 적용하며 미리 보기, NDI®를 통한 투명 배경 영상 전송, Looking Glass 출력을 제공합니다.

## 다운로드

앱을 사용하기만 하려면 [GitHub Releases](https://github.com/matsuvr/RusTuberV/releases/latest)에서 Windows x64 버전 또는 Apple Silicon Mac 테스트 버전을 다운로드하세요. 배포판에는 추론 모델과 필요한 런타임이 포함되어 있습니다. 아래 개발 환경은 소스에서 직접 빌드할 때만 필요합니다.

Windows에서는 ZIP 전체를 압축 해제한 뒤 `RusTuberV.exe`를 더블 클릭하세요. 동봉된 파일은 함께 보관하세요. Mac 버전은 macOS 14 이상이 필요하며 기술적인 문제를 직접 진단하고 해결할 수 있는 사용자를 위한 테스트 버전입니다. 제한 사항, 확인된 환경과 최초 실행 방법은 [릴리스 설명](https://github.com/matsuvr/RusTuberV/releases/latest)을 참조하세요.

## 사용 설명서

**[한국어 사용 설명서](manual/ko/README.md)**

아바타 불러오기, 카메라 선택, 구도와 표정 조정, OBS 출력을 스크린샷과 함께 설명합니다.

## 주요 기능

- VRM 0.x / 1.0 불러오기
- 웹캠을 통한 얼굴, 눈 깜빡임, 시선, 입, 상체, 팔, 손 추적
- 뼈 길이, 부모·자식 계층, 관절 축, 자유도, 가동 범위를 제약으로 적용하는 아바타 제어
- 중립 자세와 팔 보정, 좌우 반전, 구도 조정
- 키보드 표정 전환 및 모델별 키 할당
- 고급 렌더링과 모델별 설정 저장
- NDI를 통한 아바타 영상 전송
- Looking Glass Bridge를 통한 기기 검색과 출력
- 일본어, 영어, 중국어 간체, 한국어 UI

## 빌드 환경

| 항목 | 요구 사항 |
| --- | --- |
| Rust | 1.99 이상. `rust-toolchain.toml`은 stable 지정 |
| Windows | x86_64 MSVC 타깃, Visual Studio Build Tools의 C++ 빌드 도구와 Windows SDK |
| macOS | Apple Silicon과 Xcode Command Line Tools |
| NDI 출력을 포함하는 빌드 | NDI Standard SDK와 Clang / libclang |
| 렌더링 | Bevy 렌더링 백엔드를 지원하는 GPU와 드라이버 |
| 추적 | 웹캠과 운영체제의 카메라 사용 권한 |

카메라 구현은 Windows에서 Media Foundation을, macOS에서 AVFoundation을 사용합니다.

데스크톱 빌드는 기본적으로 `ndi-output`을 활성화합니다. NDI가 필요 없다면 아래의 `--no-default-features` 옵션으로 NDI SDK 의존성을 제외할 수 있습니다.

## 소스 가져오기

```sh
git clone https://github.com/matsuvr/RusTuberV.git
cd RusTuberV
```

의존성 버전은 `Cargo.lock`으로 고정합니다. 빌드할 때 `--locked`를 지정하세요.

## 추론 모델 준비

다음 세 파일을 `assets/models/`에 배치합니다. 모델 바이너리는 Git 관리 대상에서 제외되지만 빌드 시 실행 파일에 포함되므로 빌드 전에 준비해야 합니다.

| 파일 | 용도 |
| --- | --- |
| `face_landmarker.task` | 얼굴, 표정, 시선 추론 |
| `pose_landmarker_full.task` | 상체와 팔 자세 추론 |
| `hand_landmarker.task` | 손과 손가락 추론 |

Windows에서는 저장소 루트에서 다음 PowerShell 명령을 실행합니다.

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "assets/models/face_landmarker.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "assets/models/pose_landmarker_full.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "assets/models/hand_landmarker.task"
```

macOS에서는 다음을 실행합니다.

```sh
curl -fL -o assets/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -fL -o assets/models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -fL -o assets/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

승인된 SHA-256과 다운로드 출처는 [모델 매니페스트](assets/models/manifest.toml)에 기록되어 있습니다. 앱은 모델을 불러올 때 해시를 확인합니다. 상위 프로젝트의 `latest` 파일이 바뀌어 해시가 일치하지 않으면 매니페스트에 해당하는 모델 버전이 필요합니다.

네이티브 라이브러리가 포함되지 않은 소스 빌드에서는 MediaPipe 네이티브 라이브러리를 처음 추론을 시작할 때 공식 PyPI wheel에서 다운로드하여 캐시합니다. 최초 다운로드에는 네트워크 연결이 필요합니다. 라이브러리를 직접 준비한다면 `MEDIAPIPE_LIB`에 해당 플랫폼의 MediaPipe 0.10.35 라이브러리 경로를 지정합니다. 자세한 내용은 [mediapipe-rs README](vendor/mediapipe-rs/README.md)를 참조하세요.

## Windows 빌드

### NDI 출력을 포함하는 기본 빌드

[NDI SDK 공식 다운로드 페이지](https://ndi.video/for-developers/ndi-sdk/)에서 Windows용 Standard SDK를 설치합니다. NDI 바인딩 생성에는 Clang / libclang을 사용합니다.

다음 예시는 SDK와 LLVM의 기본 설치 경로를 사용합니다. 다른 위치에 설치했다면 경로를 바꾸세요.

```powershell
$env:NDI_SDK_DIR = "C:\Program Files\NDI\NDI 6 SDK"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
cargo build --release --locked -p vtuber-desktop -j 1
.\target\release\RusTuberV.exe
```

빌드 스크립트는 SDK의 임포트 라이브러리와 일치하는 NDI 런타임 DLL을 실행 파일 옆에 복사합니다. DLL이 SDK 밖에 있다면 빌드 전에 `NDI_RUNTIME_DLL`에 파일 경로를 지정합니다.

### NDI 출력 없는 빌드

```powershell
cargo build --release --locked -p vtuber-desktop --no-default-features -j 1
.\target\release\RusTuberV.exe
```

이 구성에서는 NDI SDK와 NDI 바인딩용 libclang이 필요 없습니다. 추론 모델과 MediaPipe 네이티브 라이브러리는 필요합니다.

## macOS 빌드

NDI 출력을 포함하려면 macOS용 Standard SDK를 설치하고 위치를 지정합니다.

```sh
export NDI_SDK_DIR="/Library/NDI SDK for Apple"
cargo build --release --locked -p vtuber-desktop -j 1
./target/release/RusTuberV
```

빌드 환경에서 Clang / libclang을 찾을 수 있어야 합니다. 기본 검색으로 찾지 못하면 `LIBCLANG_PATH`에 라이브러리 디렉터리를 지정합니다.

로컬 실행용 `.app`을 만들려면 같은 환경에서 다음을 실행합니다.

```sh
bash tools/build-macos.sh
open target/release/RusTuberV.app
```

[빌드 스크립트](tools/build-macos.sh)는 아이콘, 모델 매니페스트, 추적 설정을 배치하고 임시 서명을 적용합니다. NDI 라이브러리는 설치된 SDK를 참조합니다. 이 `.app`에는 NDI 런타임 재배포용 패키징이나 Apple 공증이 포함되지 않습니다.

NDI 출력 없이 빌드하려면 다음을 실행합니다.

```sh
cargo build --release --locked -p vtuber-desktop --no-default-features -j 1
./target/release/RusTuberV
```

## 시작 옵션

시작할 때 VRM 파일을 지정할 수 있습니다.

```powershell
.\target\release\RusTuberV.exe --model "C:\Avatars\avatar.vrm"
```

```sh
./target/release/RusTuberV --model /path/to/avatar.vrm
```

지정하지 않으면 앱의 **VRM 불러오기…** 버튼이나 파일 끌어 놓기로 불러옵니다. 카메라 선택과 방송 설정은 [사용 설명서](manual/ko/README.md)를 참조하세요.

## 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| `apps/desktop` | 데스크톱 실행 파일과 OS별 시작 처리 |
| `crates/vtuber-app` | UI, 설정, 모델 가져오기, 처리 조율 |
| `crates/vtuber-core` | 공통 데이터 타입과 프레임 전달 |
| `crates/vtuber-camera` | 카메라 입력 |
| `crates/vtuber-inference` | 추론과 MediaPipe 네이티브 경계 |
| `crates/vtuber-tracking` | 관측 처리와 아바타 제어 값 변환 |
| `crates/vtuber-avatar` | VRM 렌더링, 표정, 골격과 관절 제어 |
| `crates/vtuber-ndi` | NDI 전송의 네이티브 경계 |
| `crates/vtuber-platform` | OS별 기능 |
| `tools/xtask` | 개발 명령과 배포 패키징 |
| `manual/ja` | 기본 일본어 사용 설명서 |

애플리케이션 Rust 코드는 `#![forbid(unsafe_code)]`를 사용하며 필요한 네이티브 처리는 바인딩 경계에 격리합니다. 개발 제약은 [AGENTS.md](AGENTS.md)를 참조하세요.

## 배포판 만들기

배포 ZIP 생성과 확인은 [패키징 절차](tools/RELEASING.md)(일본어)를 참조하세요. Windows 빌드·패키징과 Mac 런타임 포함, 서명, ZIP 생성 절차를 설명합니다.

## 라이선스

이 프로젝트의 코드는 [MIT License](LICENSE)로 공개합니다.

의존 라이브러리, 폰트, 추론 모델, 참조 데이터에는 각각의 라이선스가 적용됩니다. 자세한 내용은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참조하세요.

NDI SDK와 런타임은 별도의 라이선스 조건을 가진 구성 요소입니다. NDI 런타임을 포함하는 배포물을 만들 때는 사용한 SDK의 라이선스와 고지 문서를 함께 배포하고 [NDI SDK 사용 및 재배포 조건](https://docs.ndi.video/all/developing-with-ndi/sdk/licensing)을 확인하세요. Windows 패키지는 `cargo xtask ndi package`로 만들 수 있습니다.

아바타 VRM 파일, 텍스처, 방송 소재는 각 권리자가 정한 조건에 따라 사용하세요. 애플리케이션의 MIT 라이선스는 해당 소재의 사용 허가를 부여하지 않습니다.

NDI® is a registered trademark of Vizrt NDI AB. NDI 관련 정보는 [NDI 공식 사이트](https://ndi.video/)를 참조하세요.

## 면책 사항

이 소프트웨어는 MIT License에 따라 있는 그대로 제공됩니다. 동작, 추론과 추적의 정확도, 성능, 특정 하드웨어나 방송 환경과의 호환성을 보증하지 않습니다.

사용으로 인한 손해, 방송·녹화 중단, 데이터 손실 및 기타 결과에 대해 저자와 저작권자는 MIT License에 정한 범위에서 책임을 지지 않습니다.

방송·녹화 소재의 권리와 얼굴이나 실내 영상이 방송에 포함되는지 여부는 사용자가 확인해야 합니다.
