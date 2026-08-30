# NDI® output — 使い方

NDI® is a registered trademark of Vizrt NDI AB.
公式サイト: https://ndi.video/

このアプリケーションは Live 画面の NDI Output から、背景透過のアバター映像を
同一 LAN 上の NDI receiver（OBS Studio + DistroAV など）へ送ります。
音声は送りません。

## 1. この ZIP に含まれているもの

Windows x64 release ZIP には、ライセンス上同梱可能な範囲で次を入れています。

- `vtuber-desktop.exe`
- `Processing.NDI.Lib.x64.dll`（NDI Standard SDK の application-local runtime）
- `NDI_SDK_LICENSE_AGREEMENT.pdf`（exact SDK License Agreement）
- `Processing.NDI.Lib.Licenses.txt`（runtime 同梱時に SDK が要求する 3rd-party notices）
- `THIRD_PARTY_NOTICES.md`
- `NDI_RUNTIME_MANIFEST.txt`
- MediaPipe face task と license（`assets/models/`）

NDI Tools、NDI Advanced SDK、HX codec、SDK header / import library は含みません。
runtime DLL は application フォルダへ置き、System32 や PATH へは入れません。

## 2. 起動

1. ZIP を任意のフォルダへ展開する。
2. `vtuber-desktop.exe` を起動する。
3. VRM アバターを import し、Ready になるまで待つ。
4. Live 画面の **NDI® Output** で Start を押す。
5. 既定の source 名は `RusTuberV` です。

既定の映像 profile は 1920x1080 / 60fps / straight-alpha BGRA です。

## 3. sender 側に追加導入が必要な場合

この ZIP は Standard SDK の x64 runtime DLL を application-local に同梱します。
通常は sender 側で NDI SDK / NDI Tools / 別途 Runtime installer は不要です。

Start 時に `NDI_RUNTIME_NOT_FOUND` と出る場合:

- `Processing.NDI.Lib.x64.dll` が `vtuber-desktop.exe` と同じフォルダにあるか確認する。
- それでも失敗する場合のみ、公式の NDI Runtime を導入する。
  - https://ndi.video/for-developers/
  - redistributable の案内: http://ndi.link/NDIRedistV6
- 非公式 mirror や出所不明の DLL は使わない。

## 4. OBS Studio + DistroAV で受信する

OBS / DistroAV / receiver 側 NDI Runtime はこの ZIP から再配布しません。
receiver マシンで各自導入してください。

1. [OBS Studio](https://obsproject.com/) を入れる。DistroAV は OBS 31.1.1 以上を要求します。
2. [DistroAV](https://github.com/DistroAV/DistroAV) を公式手順で入れる。
   Windows の一例: `winget install --exact --id DistroAV.DistroAV`
3. DistroAV は NDI Runtime 6.3 以上を要求します。receiver 側に未導入なら公式 Runtime を入れる。
4. OBS を再起動する。
5. ソース追加 → **NDI Source**。
6. `RusTuberV`（または `PC名 (RusTuberV)`）を選ぶ。
7. クロマキーは使わず、alpha 付きソースとして合成する。

version 番号は公式情報が将来変わる前提です。導入前に DistroAV README を再確認してください。

## 5. トラブルシュート

- source が見えない: 同一 LAN か、ファイアウォールが NDI discovery を落としていないかを確認する。
  このアプリはファイアウォール規則を勝手に追加しません。
- 映像が止まる / 古い: sender は latest-frame 置換です。receiver が遅くてもアプリ本体は待ちません。
- Stop 後: Live の Stop で sender を止めます。アプリが hang / panic しないことが正常です。
- NDI Tools が必要、ということはありません。見たい場合は https://ndi.video/tools/ から各自入手してください。
