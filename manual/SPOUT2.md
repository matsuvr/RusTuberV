# Spout2でOBSに映す（Windows）

Spout2は、同じWindows PCで動いているOBSなどへ、背景が透明なアバター映像を送ります。プレビューで調整した大きさ・位置・向きが出力にも反映されます。NDIとそれぞれ開始・停止でき、同時に使えます。Macでの映像送信は[NDIの手順](ja/README.md#ndiで送るwindowsmac)を参照してください。

[English](#english) · [简体中文](#简体中文) · [한국어](#한국어)

## 準備

OBSを閉じ、[公式配布ページ](https://github.com/Off-World-Live/obs-spout2-plugin/releases)からOBSのバージョンに対応したOBS Spout2 pluginをインストールして、OBSを開き直します。RusTuberVのSpout2送信機能はWindows版に同梱しています。

OBSとRusTuberVは同じGPUで実行します。GPUが複数あるPCでは、Windowsの「設定」→「システム」→「ディスプレイ」→「グラフィックス」で両方のアプリに同じGPUを指定します。

## 送信する

1. RusTuberVでアバターを読み込み、「NDI・Spout2出力」を開きます。
2. 「Spout2送信を開始」を押し、「RusTuberV: 送信中」と表示されることを確認します。
3. OBSの「ソース」で「Spout2 Capture」を追加し、送信元に `RusTuberV` を選びます。
4. Spout2 Captureの「Composite Mode」で「Premultiplied Alpha」を選びます。映像は1920×1080、乗算済みアルファです。
5. OBSの「ソース」でアバターを背景画像やゲーム画面より上に並べます。声はOBSでマイクを設定します。

## 終了と再開

配信・録画を終えるときは、OBSで停止してからRusTuberVの「Spout2送信を停止」を押します。アバターを閉じた場合も送信が停止します。

送信中にアバターを変更した場合やPCのスリープから戻った後は、アバターの準備ができると送信を再開します。エラーが表示された場合は、内容を確認してから改めて開始してください。

## 受信と透過の確認

| 確認する項目 | 操作 |
| --- | --- |
| Spout2 Captureの追加 | OBSとプラグインの対応バージョンを確認してインストールし、OBSを開き直す |
| GPUの選択 | OBSとRusTuberVを同じGPUで実行する |
| 背景の透過 | Composite ModeをPremultiplied Alphaに設定する |
| 送信元の選択 | RusTuberVの送信中表示を確認し、OBSで使用する送信元を選ぶ。複数起動時は名前に番号が付くことがある |

## English

Spout2 sends transparent avatar video to OBS on the same Windows PC. Preview framing applies to the output. NDI and Spout2 have separate start and stop controls and can run together. For Mac, follow the [NDI guide](en/README.md#send-with-ndi-windows-and-mac).

### Setup

Close OBS, install the [Spout2 plugin matching your OBS version](https://github.com/Off-World-Live/obs-spout2-plugin/releases), and reopen OBS. The Windows RusTuberV package includes the sender. Run both apps on the same GPU; on computers with multiple GPUs, select the same GPU for each app in Windows Settings → System → Display → Graphics.

### Send video

1. Load an avatar in RusTuberV and open **NDI & Spout2 output**.
2. Click **Start Spout2** and check for **RusTuberV: Sending**.
3. Add **Spout2 Capture** to OBS and select `RusTuberV` as the sender.
4. Set **Composite Mode** to **Premultiplied Alpha** for transparent 1920×1080 video.
5. Place the avatar above background sources in OBS and configure your microphone there.

Stop streaming or recording in OBS, then click **Stop Spout2** in RusTuberV. Closing the avatar stops transmission. During an active session, transmission resumes when the avatar is ready after a model change or waking the PC. For a sender error, review the message and start transmission again.

For reception, check plugin compatibility, the selected GPU, the sender, and Composite Mode. Multiple RusTuberV instances may have numbered sender names; select the intended instance in OBS.

## 简体中文

Spout2向同一台Windows电脑上的OBS发送透明背景虚拟形象视频。预览中的构图调整会反映到输出中。NDI与Spout2可分别开始、停止，也可以同时使用。Mac的发送操作请参阅[NDI手册](zh-CN/README.md#通过ndi发送windowsmac)。

### 准备

关闭OBS，安装[与OBS版本对应的Spout2插件](https://github.com/Off-World-Live/obs-spout2-plugin/releases)，再打开OBS。Windows版RusTuberV包含Spout2发送功能。两款应用应使用同一块GPU；多GPU电脑可在Windows的“设置”→“系统”→“显示”→“图形”中为两款应用指定同一块GPU。

### 发送视频

1. 在RusTuberV中加载虚拟形象，打开“NDI与Spout2输出”。
2. 点击“开始Spout2发送”，确认显示“RusTuberV: 发送中”。
3. 在OBS中添加“Spout2 Capture”，选择发送源`RusTuberV`。
4. 将“Composite Mode”设为“Premultiplied Alpha”，以透明背景接收1920×1080视频。
5. 在OBS中将虚拟形象放在背景源上方，并设置麦克风。

结束时，先在OBS中停止直播或录制，再在RusTuberV中点击“停止Spout2发送”。关闭虚拟形象也会停止发送。发送期间更换模型或电脑从睡眠恢复后，虚拟形象准备就绪时会恢复发送。出现发送错误时，请确认提示后重新开始。

接收时请检查插件版本、GPU、发送源和Composite Mode。启动多个RusTuberV时，发送源名称可能带编号，请在OBS中选择所需的实例。

## 한국어

Spout2는 같은 Windows PC의 OBS로 투명 배경 아바타 영상을 보냅니다. 미리 보기의 구도 조정이 출력에도 반영됩니다. NDI와 Spout2는 각각 시작·중지할 수 있고 동시에 사용할 수도 있습니다. Mac에서는 [NDI 설명서](ko/README.md#ndi로-전송windowsmac)를 참조하세요.

### 준비

OBS를 닫고 [OBS 버전에 맞는 Spout2 플러그인](https://github.com/Off-World-Live/obs-spout2-plugin/releases)을 설치한 뒤 OBS를 다시 엽니다. Windows용 RusTuberV에는 송신 기능이 포함되어 있습니다. 두 앱은 같은 GPU로 실행하세요. 여러 GPU가 있는 PC에서는 Windows 설정 → 시스템 → 디스플레이 → 그래픽에서 두 앱에 같은 GPU를 지정합니다.

### 영상 전송

1. RusTuberV에서 아바타를 불러오고 **NDI 및 Spout2 출력**을 엽니다.
2. **Spout2 전송 시작**을 누르고 **RusTuberV: 전송 중**을 확인합니다.
3. OBS에 **Spout2 Capture**를 추가하고 송신자로 `RusTuberV`를 선택합니다.
4. **Composite Mode**를 **Premultiplied Alpha**로 설정해 투명 배경 1920×1080 영상을 받습니다.
5. OBS에서 아바타를 배경 소스보다 위에 배치하고 마이크를 설정합니다.

끝낼 때는 OBS에서 방송·녹화를 중지한 뒤 RusTuberV에서 **Spout2 전송 중지**를 누릅니다. 아바타를 닫아도 전송이 중지됩니다. 전송 중 모델을 바꾸거나 PC가 절전에서 돌아오면 아바타가 준비된 뒤 전송이 재개됩니다. 송신 오류가 표시되면 내용을 확인한 뒤 다시 시작하세요.

수신할 때는 플러그인 버전, GPU, 송신자와 Composite Mode를 확인합니다. RusTuberV를 여러 개 실행하면 송신자 이름에 번호가 붙을 수 있으므로 OBS에서 사용할 항목을 선택하세요.
