# RusTuberV

[日本語](README.md) · [English](README.en.md) · **简体中文** · [한국어](README.ko.md)

RusTuberV 是使用 Rust 和 Bevy 开发的摄像头驱动 VRM 虚拟形象应用。它将 MediaPipe 的脸部、姿势和手部推理结果应用于虚拟形象，提供预览、通过 NDI® 发送透明背景视频以及 Looking Glass 输出。

## 使用手册

**[简体中文使用手册](manual/zh-CN/README.md)**

手册通过截图介绍虚拟形象加载、摄像头选择、构图与表情调整以及 OBS 输出。

## 主要功能

- 加载 VRM 0.x / 1.0
- 通过摄像头跟踪脸部、眨眼、视线、嘴部、上半身、手臂和手部
- 以骨骼长度、父子层级、关节轴、自由度和活动范围为约束控制虚拟形象
- 自然姿势与手臂校准、镜像和构图调整
- 使用键盘切换表情，并为每个模型保存按键绑定
- 增强显示与模型独立设置
- 通过 NDI 发送虚拟形象视频
- 通过 Looking Glass Bridge 检测设备并输出
- 日语、英语、简体中文和韩语界面

## 构建环境

| 项目 | 要求 |
| --- | --- |
| Rust | 1.99 或更高版本；`rust-toolchain.toml` 指定 stable |
| Windows | x86_64 MSVC 目标、Visual Studio Build Tools 的 C++ 构建工具及 Windows SDK |
| macOS | Apple Silicon 和 Xcode Command Line Tools |
| 包含 NDI 输出的构建 | NDI Standard SDK 和 Clang / libclang |
| 渲染 | 支持 Bevy 渲染后端的 GPU 和驱动程序 |
| 跟踪 | 摄像头及操作系统的摄像头使用权限 |

摄像头实现使用 Windows 的 Media Foundation 和 macOS 的 AVFoundation。

桌面构建默认启用 `ndi-output`。无需 NDI 时，可按下文使用 `--no-default-features` 移除对 NDI SDK 的依赖。

## 获取源码

```sh
git clone https://github.com/matsuvr/RusTuberV.git
cd RusTuberV
```

依赖版本由 `Cargo.lock` 固定。构建时请使用 `--locked`。

## 准备推理模型

将以下三个文件放入 `assets/models/`。模型二进制文件不纳入 Git 管理，但会在构建时嵌入可执行文件，因此必须在构建前准备好。

| 文件 | 用途 |
| --- | --- |
| `face_landmarker.task` | 脸部、表情和视线推理 |
| `pose_landmarker_full.task` | 上半身与手臂姿势推理 |
| `hand_landmarker.task` | 手部与手指推理 |

在 Windows 上，从仓库根目录运行以下 PowerShell 命令。

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "assets/models/face_landmarker.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "assets/models/pose_landmarker_full.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "assets/models/hand_landmarker.task"
```

在 macOS 上运行：

```sh
curl -fL -o assets/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -fL -o assets/models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -fL -o assets/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

认可的 SHA-256 和下载来源记录在[模型清单](assets/models/manifest.toml)中。应用加载模型时会校验哈希。如果上游 `latest` 文件更新后哈希不再匹配，请使用与清单对应的模型版本。

首次启动推理时，MediaPipe 原生库会从官方 PyPI wheel 下载并缓存。首次下载需要网络连接。如自行准备原生库，请将 `MEDIAPIPE_LIB` 设置为对应平台的 MediaPipe 0.10.35 库文件路径。详情见 [mediapipe-rs README](vendor/mediapipe-rs/README.md)。

## 在 Windows 上构建

### 包含 NDI 输出的默认构建

从 [NDI SDK 官方下载页面](https://ndi.video/for-developers/ndi-sdk/)安装 Windows Standard SDK。生成 NDI 绑定需要 Clang / libclang。

以下示例使用 SDK 和 LLVM 的标准安装路径。若安装在其他位置，请修改路径。

```powershell
$env:NDI_SDK_DIR = "C:\Program Files\NDI\NDI 6 SDK"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
cargo build --release --locked -p vtuber-desktop -j 1
.\target\release\RusTuberV.exe
```

构建脚本会将与 SDK 导入库匹配的 NDI 运行时 DLL 复制到可执行文件旁。若 DLL 位于 SDK 目录之外，请在构建前将 `NDI_RUNTIME_DLL` 设置为该文件的路径。

### 不包含 NDI 输出的构建

```powershell
cargo build --release --locked -p vtuber-desktop --no-default-features -j 1
.\target\release\RusTuberV.exe
```

此配置无需 NDI SDK 或用于生成 NDI 绑定的 libclang。仍须准备推理模型及 MediaPipe 原生库。

## 在 macOS 上构建

需要 NDI 输出时，安装 macOS Standard SDK 并指定其目录。

```sh
export NDI_SDK_DIR="/Library/NDI SDK for Apple"
cargo build --release --locked -p vtuber-desktop -j 1
./target/release/RusTuberV
```

请确保构建环境可以找到 Clang / libclang。如果默认搜索无法找到，请通过 `LIBCLANG_PATH` 指定库目录。

如需生成本地运行用的 `.app`，在同一环境中执行：

```sh
bash tools/build-macos.sh
open target/release/RusTuberV.app
```

[构建脚本](tools/build-macos.sh)会放置图标、模型清单和跟踪设置，并进行临时签名。NDI 库引用已安装的 SDK。此 `.app` 不包含 NDI 运行时再分发打包或 Apple 公证。

无需 NDI 输出时执行：

```sh
cargo build --release --locked -p vtuber-desktop --no-default-features -j 1
./target/release/RusTuberV
```

## 启动选项

可在启动时指定 VRM 文件。

```powershell
.\target\release\RusTuberV.exe --model "C:\Avatars\avatar.vrm"
```

```sh
./target/release/RusTuberV --model /path/to/avatar.vrm
```

未指定时，可通过应用中的“选择VRM…”或拖入文件进行加载。摄像头选择和直播设置请参阅[使用手册](manual/zh-CN/README.md)。

## 项目结构

| 路径 | 职责 |
| --- | --- |
| `apps/desktop` | 桌面可执行文件和各操作系统的启动处理 |
| `crates/vtuber-app` | 界面、设置、模型导入和流程协调 |
| `crates/vtuber-core` | 公共数据类型和帧传递 |
| `crates/vtuber-camera` | 摄像头采集 |
| `crates/vtuber-inference` | 推理和 MediaPipe 原生边界 |
| `crates/vtuber-tracking` | 观测处理及虚拟形象控制值转换 |
| `crates/vtuber-avatar` | VRM 渲染、表情、骨骼和关节控制 |
| `crates/vtuber-ndi` | NDI 发送的原生边界 |
| `crates/vtuber-platform` | 操作系统专属功能 |
| `tools/xtask` | 开发命令和分发打包 |
| `manual/ja` | 默认的日语使用手册 |

应用 Rust 代码使用 `#![forbid(unsafe_code)]`，必要的原生操作隔离在绑定边界内。开发约束请参阅 [AGENTS.md](AGENTS.md)。

## 许可证

本项目代码以 [MIT License](LICENSE) 发布。

依赖库、字体、推理模型和参考数据分别适用各自的许可证。详情见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

NDI SDK 和运行时是具有独立许可条款的组件。制作包含 NDI 运行时的分发包时，请附带所用 SDK 的许可与声明文件，并遵守 [NDI SDK 使用及再分发条款](https://docs.ndi.video/all/developing-with-ndi/sdk/licensing)。Windows 分发包可使用 `cargo xtask ndi package` 创建。

虚拟形象 VRM 文件、纹理和直播素材的使用须遵守各权利人的规定。应用的 MIT 许可证并不授予这些素材的使用权。

NDI® is a registered trademark of Vizrt NDI AB. NDI 相关信息请参阅 [NDI 官方网站](https://ndi.video/)。

## 免责声明

本软件依据 MIT License 按现状提供。不保证软件运行、推理与跟踪精度、性能，或与特定硬件及直播环境的兼容性。

对于使用软件产生的损害、直播或录制中断、数据丢失及其他后果，作者和版权人在 MIT License 规定的范围内不承担责任。

用户须自行确认直播与录制素材的权利，以及自己的脸部或室内画面是否会出现在直播中。
