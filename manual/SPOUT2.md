# Spout2でOBSに映す（Windows）

Spout2は、同じWindows PCで動いているOBSなどへ、背景が透明なアバター映像を送ります。カメラ映像や操作画面は含まれません。NDIとそれぞれ独立して開始・停止できます。MacではNDIを使用してください。

## 準備

OBSに[OBS Spout2 plugin](https://github.com/Off-World-Live/obs-spout2-plugin/releases)を導入し、OBSを起動し直します。OBSのバージョンに対応したプラグインを選んでください。OBS 32には1.13.0、OBS 33には1.14.0以降が対応します。RusTuberV側へのSpout SDKやDLLの追加インストールは不要です。

## 送信する

1. RusTuberVでアバターを読み込み、「出力」を開きます。
2. 「Spout2送信を開始」を押します。
3. OBSの「ソース」で「Spout2 Capture」を追加し、送信元に `RusTuberV` を選びます。
4. 背景を透過させるには、Spout2 Captureの「Composite Mode」で **Premultiplied Alpha** を選びます。映像は1920×1080、乗算済みアルファです。

終了するときは「Spout2送信を停止」を押します。アバターを閉じた場合も停止します。アバターの変更やPCのスリープから戻った後は、アバターの準備ができると送信を再開します。エラーが表示された場合は、内容を確認してから改めて開始してください。

## 映らないとき

- OBSとRusTuberVを同じGPUで実行してください。GPUが複数あるPCではWindowsの「グラフィックス」設定で確認できます。
- OBSに「Spout2 Capture」がない場合は、プラグインがOBSのバージョンに対応しているか確認してください。
- 背景が黒く塗りつぶされる場合は、Composite Modeを **Premultiplied Alpha** に変更してください。
- 複数のRusTuberVを起動すると、送信元の名前に番号が付くことがあります。OBSで使用する送信元を選び直してください。

## English

Install the [Spout2 plugin matching your OBS version](https://github.com/Off-World-Live/obs-spout2-plugin/releases) and restart OBS. In RusTuberV, load an avatar, open **Output**, and select **Start Spout2**. Add **Spout2 Capture** in OBS, select `RusTuberV`, and set **Composite Mode** to **Premultiplied Alpha** for transparency. Output is 1920×1080 BGRA with premultiplied alpha. Run both applications on the same GPU. NDI and Spout2 can be started and stopped independently.
