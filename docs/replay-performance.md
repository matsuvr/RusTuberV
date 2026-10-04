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
- `run-N.log`: 起動・モデル・推論エラー等。

`fps` / `frame_ms` は画面内のBevy診断と同じ `Time<Real>` のフレーム周期。
描画側から渡された時計を使い、平滑化やVirtual Timeの上限で長いフレームを隠さない。
`wall_fps` / `main_interval_ms` はLastの測定地点同士の実時間間隔も保持する。
通常は本番と同じVSyncを使う。`-Uncapped` は測定ウィンドウだけVSyncを解除し、
60Hzの表示上限の背後にどれだけ処理余裕があるかを調べる。結果にはモードを保存し、
VSyncありと上限解除を直接改善率として比較しない。本番のVSync設定は変えない。
`main_world_ms` はFirstからLastの計測地点までで、GPUの実描画・表示完了時刻ではない。
推論時間は新しい完了結果を観測したときだけ集計し、同じ値を描画tickごとに重複計上しない。
描画が遅く複数結果が飛ばされたとき、その間の各推論時間は復元しない。
推論頻度はworkerの処理済みカウンターから求める。
CPU使用率は既存のBevy診断値で、ホスト全体の論理CPU数を分母とする。
3個へ制限した予算の使用率ではない。

`arm_first_apply_ms` は、その入力番号の姿勢が初めて制約付き表示経路へ採用された時点の
入力年齢。`arm_displayed_age_ms` は、各描画tickで使用中の入力番号の年齢。
補間の進行中にも採用目標の番号を持つため、各画素がその入力に完全一致する時間ではない。
入力を止めてfpsだけ上げる改善を識別するため、fpsと一緒に比較する。
骨長・関節自由度・可動域・接触制約と、交差からの復帰も既存の検査と表示で確認する。

物理カメラのセンサー・ドライバー、動画のオフライン復号、NDI送信、画面発光までの
遅延はこの測定に含まれない。動画と生成ファイルはgitignore対象の `data/` に留める。

3論理CPUでの実測値、採用・撤回した軽量化、内蔵GPUを含む限界は
[上肢ADRの最終候補の実測](adr/2026-10-03-upper-limb-anatomy.md#最終候補の実測)を参照。
