# AGENTS.md

## Mission

`DESIGN.md` の実装を進める: Bevy 0.19 + `bevy_vrm1` によるフルRust・デスクトップVTuberアプリ（Windows 11 x86_64 / macOS 13+、VRM 0.x と VRM 1.0）。設計と制約の一次情報は `DESIGN.md` と `docs/adr/`。

## 純Rust

ランタイムは Rust だけで構成し、アプリクレートは `#![forbid(unsafe_code)]` とする。

ネイティブコードを許す例外は MediaPipe Tasks 0.10.35 のみ。ADR-009 が固定する `mediapipe-rs` リビジョン経由で使い、`vtuber-inference` の内側に隔離する。

## 失敗は型で伝える

プロダクションコードはデフォルトでパニックしない。CLI引数・設定・ファイルシステム・ネットワーク・外部データ・クロックなど、実行時の状況に依存して失敗しうる処理は、`Result`・`Option`・明示的な結果型・検証済みの値オブジェクトで表現し、呼び出し元へ伝播させる。`unwrap`・`expect`・`panic!`・`unreachable!`・`todo!`・unchecked indexing は、この表現の代わりには使わない。

それでもパニックがやむを得ないのは、依存に代替がなく、条件が実行時データから独立していると証明できる内部不変条件に限る。その場合は例外を最小限に保ち、呼び出し箇所に不変条件を文書化する。

テストとテスト用フィクスチャでは `unwrap` / `expect` / `panic!` を自由に使ってよい。

この方針は workspace lints（`clippy::unwrap_used` など）で機械的に確認する。検証は `cargo clippy --workspace --all-targets` をローカルで実行する。
