# AGENTS.md

- やむを得ないネイティブバインディング以外はpure Rust。バインディングは境界内に隔離し、アプリクレートは `#![forbid(unsafe_code)]` とする。
- 人体の解剖学的構造（骨長・親子階層・関節軸・自由度・可動域）を厳密な制約として完全に守り、観測追従や見た目のために破らない。
- CPU 3コアで通常描画60fps・推論30Hzを維持する。任意のLooking Glass出力は描画60fpsの対象外とする。
- プロダクションコードはpanic禁止。失敗は `Result`・`Option` 等で伝え、`unwrap`・`expect`・パニックする添字アクセス等で代用しない。
- 要件を満たす最も単純な実装を選ぶ。不要な経路は削除し、依頼のないガード・フォールバック・汎用化・検証基盤を追加しない。
- README・リリースページなどの利用者向け文書には、利用者に必要な情報だけを書く。検証結果・テスト件数・検証環境・検証の未実施や未確認事項・内部事情や言い訳は記載しない。必要な検証は実施し、結果は開発用の記録や作業報告に留める。
- サブエージェントの利用禁止。
- Windowsの標準ビルドはMSYS2 CLANG64（LLVM）とし、build・check・test・runは `tools/cargo-windows.ps1` 経由で実行する。配布ZIPは `tools/package-windows.ps1` を使う。MSVCは明示的な切り戻し時のみ使用し、手順は `tools/WINDOWS-LLVM.md` に従う。
- 追跡フィルター・関節への動きの伝播・腕IKを変更するたびに、`.\tools\cargo-windows.ps1 -CargoArguments @('test', '--locked', '-p', 'vtuber-tracking', '-p', 'vtuber-avatar', '--lib', '-j', '1')` を実行する。減衰・複数ノイズの増幅・最終骨出力の回帰テストを省略しない。
