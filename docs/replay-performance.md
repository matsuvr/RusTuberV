# 動画によるカメラ推論・表示の反復測定

Windows のリポジトリルートで実行する。FFmpeg は動画の事前変換だけに使用し、
アプリの実行時依存にはしない。PATH または WinGet の Gyan.FFmpeg が見つからない
場合は `-Ffmpeg <ffmpeg.exe>` を指定する。

```powershell
./tools/measure-replay.ps1 `
  -Video @('data/video/WIN_20261004_22_29_52_Pro.mp4',
           'data/video/WIN_20261004_22_48_07_Pro.mp4') `
  -Model data/vrm_data/Sapphy.vrm `
  -LogicalProcessors 3 -Runs 3 -Tag before
```

処理は release ビルド、入力変換、動画ごとに独立プロセスで3回の測定、結果保存まで。
人がカメラの前に立つ操作は不要。各回は初期姿勢の制約解決後に動画を開始し、
1周で自動終了する。起動に失敗した場合はタイムアウトまたは既存のエラーを返す。
コード修正後に同じコマンドを `-Tag after` で実行して比較する。
`-NoBuild` は既にビルドした実行ファイルを再使用する場合だけ指定する。
スクリプト自身がソースコードを自動改変する機能は持たない。

入力は既定カメラ要求と同じ640×360 RGB8・30fpsへ変換し、動画SHA-256を含む
ディレクトリへキャッシュする。再生が遅れた場合は動画の現在時刻へ飛び、
古いフレームを高速に消化しない。通常の capture worker と容量1の配信先へ流し、
本物のMediaPipe Face、Pose、Hand、追従、人体制約、GUI・アバター描画を動かす。
既存の利用者設定には保存せず、各回に独立した設定・管理モデルを使う。
腕追従ON、リッチ表示OFF、NDI送信OFF、通常ウィンドウ設定が比較条件となる。
表示イベントの計測が他のアプリによる遮蔽で途切れないよう、測定窓は動画の1周が
終わるまで最前面に表示する。本番のウィンドウの重なり順は変更しない。

`-LogicalProcessors 3` はBevyのプール数だけでなく、プロセス全体のCPU affinityを
起動前に3ビットへ制限し、Bevyの総スレッド予算も明示する。
WindowsでRustの`available_parallelism`がaffinity制限前の個数を返す場合にも、
大きなプールで少CPU機の測定を歪めない。BevyのIO・描画の最小スレッド数により、
OSスレッド数の合計はCPU予算と一致するとは限らない。
MediaPipeや描画ドライバーのプロセス内スレッドも同じ
CPU集合を使う。現在の許可マスクの下位から選び、実際のマスクを記録する。
`-ProcessorAffinity 0x15` のように3ビットのマスクを指定することもできる。
このi9-13900では `0x7` は2物理コアを共有する3論理CPU、`0x15` は3物理コアの
各1スレッドである。CPU番号と物理コアの対応は機種固有なので、他のPCへそのまま
当てはめない。比較前後で同じマスクを使う。
今回のPCで記録した比較を再現する場合は、上のコマンドへ
`-ProcessorAffinity 0x15` を追加する。
SMT・CPU世代・GPU性能・電力制限は別なので、廉価ノート実機との同等性は主張しない。
[Windowsのプロセス継承仕様](https://learn.microsoft.com/en-us/windows/win32/procthread/inheritance)
に従い、親のaffinityは実行後に戻す。

`data/performance/<日時>-<タグ>-<動画名>/` に保存するもの:

- `metadata.json`: 動画・VRM・実行ファイル・追従設定・モデルmanifestのSHA-256、Git状態、CPU制限。
- `working-tree.patch`: 追跡対象ファイルの未コミット差分。実行ファイルの内容を証明する
  ものではないので、修正後は再ビルドする。
- `comparison.csv`: 各回のfps、フレームp95、推論頻度、腕遅延p95。
- `run-N/summary.json`: 最初の5秒を除いた分布と入力条件。
- `run-N/frames.jsonl`: 全フレームの時間、推論回数、腕入力番号、最終関節位置・回転。
- `run-N/tracking_profile.toml`: 実行時の追従設定の写し。
- `run-N/avatar-XX.png`: 動画時刻に対応する表示画像。PNG圧縮は測定終了後に行う。
- `run-N.log` / `run-N-error.log`: 標準出力と起動・モデル・推論エラー等。

`fps` / `frame_ms` は画面内のBevy診断と同じ `Time<Real>` のフレーム周期。
描画側から渡された時計を使い、平滑化やVirtual Timeの上限で長いフレームを隠さない。
`wall_fps` / `main_interval_ms` はLastの測定地点同士の実時間間隔も保持する。
通常は本番のWindow既定値と同じ `PresentMode::Fifo` を使う。
以前の測定は `AutoVsync` だったため、対応環境では `FifoRelaxed` が選ばれ得た。
`-Uncapped` は測定ウィンドウだけVSyncを解除し、
60Hzの表示上限の背後にどれだけ処理余裕があるかを調べる。結果にはモードを保存し、
VSyncありと上限解除を直接改善率として比較しない。本番のVSync設定は変えない。
`main_world_ms` はFirstからLastの計測地点までで、GPUの実描画・表示完了時刻ではない。
推論時間は新しい完了結果を観測したときだけ集計し、同じ値を描画tickごとに重複計上しない。
描画が遅く複数結果が飛ばされたとき、その間の各推論時間は復元しない。
推論頻度はworkerの処理済みカウンターから求める。
CPU使用率は既存のBevy診断値で、ホスト全体の論理CPU数を分母とする。
3個へ制限した予算の使用率ではない。

画面に表示された間隔を調べる場合は、Windows用の
[PresentMon Console](https://github.com/GameTechDev/PresentMon/releases) の実行ファイルを
`-PresentMon <PresentMon.exe>` で指定する。インストールやアプリへの組込みは不要。
各回の `run-N-presentmon.csv` とログに、OSの表示イベントを保存する。
PresentMon自身は測定対象のCPU affinityに含めない。
`summary.json` の `video_started_unix_seconds` と `video_duration_seconds` を使って
CSVのQPC時刻を `metadata.json` の `presentmon_clock` で動画時刻へ対応付け、
同じ5秒のウォームアップを除く。動画時刻との照合はミリ秒単位で行い、
表示間隔自体にはPresentMonのQPC差分を使う。
開始日時は外部トレースとの照合専用で、アニメーションやフレーム時計には使わない。
PresentMon 2.6の `MsBetweenDisplayChange` は表示更新の間隔で、未表示は `NA`。
Bevyの `frame_ms` やPresent呼出間隔とは別に評価する。
PresentMonも画面の発光を光学測定するものではない。
`run-N/presentation-summary.json` に表示間隔・GPU稼働時間・Present呼出から表示までの時間、
25msを超える表示間隔の数を集計する。25msは60Hzで更新を1回逃した区間を数えるためで、
他のリフレッシュレートの合否基準にはしない。`display_time_seconds` と要求区間の長さ、
最初・最後の動画時刻も確認する。画面の消灯・ロック・非表示やイベント欠落で途中までしか
記録されなかった場合、その部分のfpsを動画全体の測定結果として扱わない。
既存のCSVも `tools/summarize-presentation.ps1` で再集計できる。

`arm_first_apply_ms` は、その入力番号の姿勢が初めて制約付き表示経路へ採用された時点の
入力年齢。`arm_displayed_age_ms` は、各描画tickで使用中の入力番号の年齢。
補間の進行中にも採用目標の番号を持つため、各画素がその入力に完全一致する時間ではない。
入力を止めてfpsだけ上げる改善を識別するため、fpsと一緒に比較する。
骨長・関節自由度・可動域・接触制約と、交差からの復帰も既存の検査と表示で確認する。

物理カメラのセンサー・ドライバー、動画のオフライン復号、NDI送信、画面発光までの
遅延はこの測定に含まれない。動画と生成ファイルはgitignore対象の `data/` に留める。

3論理CPUでの実測値、採用・撤回した軽量化、内蔵GPUを含む限界は
[上肢ADRの最終候補の実測](adr/2026-10-03-upper-limb-anatomy.md#最終候補の実測)を参照。

## 更新と表示の同期方針

FaceとPose/Handは専用workerで推論し、容量1のLatestSlotへ結果を公開する。
Updateは完了済みの最新結果を読む。推論本体はslotやstatusのロックを保持せず、
Updateから推論完了を待たない。未完了なら既存の保持・減衰と、制約を認定済みの
腕経路を描画tickごとに進める。新しい観測が来るたびに描画を止める構成にはしない。

描画にはDefaultPluginsのPipelinedRenderingPluginを使う。メイン側の更新と
描画スレッドが1フレームずれて並行し、受け渡し時に同期する。Window既定のFIFOは
ディスプレイの更新周期に合わせて表示するので、独自に60Hzのsleepを重ねない。
高リフレッシュレートでも同じ構成を使えるが、その周期内で処理できるかは実測が必要。
推論30Hzと描画60Hz以上は別の周期であり、推論結果を得るために毎描画tickで
推論を再実行する必要はない。

参照した公開資料と実装:

- [Bevy 0.19のPipelinedRenderingPlugin](https://github.com/bevyengine/bevy/blob/v0.19.0/crates/bevy_render/src/pipelined_rendering.rs): 更新・抽出・描画の所有権と同期点。
- [Unreal EngineのThreaded Rendering](https://dev.epicgames.com/documentation/en-us/unreal-engine/threaded-rendering-in-unreal-engine): 更新と描画を分離し、更新済みデータのコピーを渡す。
- [Unreal EngineのLow-Latency Frame Syncing](https://dev.epicgames.com/documentation/en-us/unreal-engine/low-latency-frame-syncing-in-unreal-engine): 表示に対する先行量は、遅延と処理変動への余裕の交換条件。
- [UnityのtargetFrameRate](https://docs.unity3d.com/ScriptReference/Application-targetFrameRate.html): デスクトップの滑らかな表示にはソフトウェアの時間制限よりハードウェア同期を推奨。
- [Unityの公開JobHandleソース](https://github.com/Unity-Technologies/UnityCsReference/blob/master/Modules/ManagedKernel/Managed/Jobs/JobHandle.bindings.cs): 完了状態の確認と、完了まで待つ操作を区別する。ソースを移植せず、既存のBevy/Rustの仕組みを使う。

## OS表示イベントでの再測定（2026-10-05）

main `7977065` を基準に、上記の2動画を各2回ずつ再生した。
Windows 11 / i9-13900 / RTX 4090、Vulkan、Sapphy、1920×1080、60Hz、
3論理CPU（affinity `0x15`）、Bevy予算3、Fifo、リッチOFF・NDI OFF。
PresentMon 2.6を別プロセスで実行し、動画の先頭5秒を除く全区間を取得した。
通常は各55.6秒、交差は各26.57秒。これは物理カメラを含む測定ではない。

| 動画（各2回） | 表示fps | 表示間隔p95 | 表示間隔p99 | 25ms超の表示間隔 | Bevy時計p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 通常 | 59.784 / 59.802 | 16.703 / 16.706ms | 16.730 / 16.731ms | 12 / 11回 | 22.096 / 21.863ms |
| 腕交差 | 59.774 / 59.774 | 16.707 / 16.696ms | 16.748 / 16.729ms | 6 / 6回 | 21.656 / 21.863ms |

Bevyの時計だけから「p95で約22ms画面が止まる」とは判断できない。
一方、9,825回の表示更新中35回（約0.36%）に約33msの間隔があり、
全更新で60fps維持は未達。4走行ともティアリング許可は0件、未表示フレームは0件だった。
Faceは27.57–29.92Hz、Pose/Handは29.92–30.02Hzで継続している。
人体の制約・推論モデル・補間方法はこの比較では変更していない。
記録した関節値は全て有限で、骨長の変動幅は最大 `3.05e-7 m`。
この数値確認だけで全画素や全関節制約の正しさを新たに証明したとは扱わない。

長い表示間隔に対応するPresent呼出間隔は約33–46msだった。
通常動画の14–18秒、23秒台など、交差動画の6.997秒、12.534秒、24秒台などに
現れたが、反復ごとに同じ時刻には発生しない。1件は10秒の測定用画像取得と重なった。
該当フレームのPresent API内は約0.12–0.67ms、GPU稼働は約0.29–2.87msで、
「Present API内で33ms待った」「腕交差だけでGPU描画が重くなった」とは説明できない。
CPU側の処理・スケジューリング・描画同期を区別するには追加のCPUトレースが必要で、
この記録だけから主因を断定しない。GPU値はOSトレース由来の参考値である。

標準エンジンの同期方法に照合して、次の2案も各4走行ずつ試した。

| 試行 | 通常: 25ms超（各回） | 交差: 25ms超（各回） | 判断 |
| --- | ---: | ---: | --- |
| 現行 | 12 / 11 | 6 / 6 | 基準 |
| Renderの外側スケジュールも呼出元で実行 | 13 / 9 | 5 / 2 | 通常で悪化もあり、採用せず |
| 最大フレーム先行量のhintを2→3 | 10 / 9 | 8 / 2 | 交差で悪化もあり、採用せず |

どちらも一貫した改善を確認できなかったので撤回した。後者は
[wgpuのsurface設定](https://docs.rs/wgpu/latest/wgpu/type.SurfaceConfiguration.html)
の標準hintであり、実際のキュー長を保証しない。今回のPresent→表示の平均は
基準15.97–16.09ms、hint=3で15.92–16.17msであり、実測上は1フレーム増えていない。
本番の同期設定や人体コードは変えず、残す変更は測定のFifo統一と表示イベントの採取・集計。
高リフレッシュレート、廉価ノート、物理カメラ、macOS、NDI送信の性能は未確認。

完全なトレースを取得できたローカル記録（`data/performance/`）:

- 基準: `20261005-073010-display-final-*` / `20261005-073227-display-final-*`
- 外側Render試行: `20261005-073739-render-tail-inline-*` / `20261005-073956-render-tail-inline-*`
- 先行量3試行: `20261005-074924-fifo-latency3-*` / `20261005-075140-fifo-latency3-*`

途中で記録が切れた予備試行は上の集計に含めない。強制終了した計測器のETWセッションが
残っていたため、以後は各走行に専用のセッション名を付け、終了時にそのセッションだけを
明示的に停止してCSVを書き切る。
