# ローカルの push 前に Jev で差分を確認する

`jev-lint` 0.7.0 で push 対象の Rust の変更行について、名前・コメント・テスト名などを確認します。コミット文と `AGENTS.md` への適合も確認します。自前コードの `apps/`、`crates/`、`tools/` を対象にし、`vendor/` は対象外です。結果は警告として表示し、指摘だけでは push を止めません。GitHub Actions では実行しません。

## 一度だけ設定

Node.js 24 以上と `npx` が必要です。リポジトリのルートで実行します。

```sh
git config --local core.hooksPath .githooks
```

この設定は自分のクローンの `.git/config` にだけ保存されます。既に `core.hooksPath` を使っている場合は、この設定を上書きせず、既存の `pre-push` フックから `.githooks/pre-push` を呼び出してください。Git Bash を使う Windows と macOS の Git で動くシェルフックです。

## API キー

[TypeSafe AI](https://typesafe.ai/) で取得したキーを、push を起動するプロセスの環境変数に設定します。キーはリポジトリの設定ファイルに書きません。

macOS / Linux / Git Bash のシェルの場合:

```sh
export TYPESAFE_API_KEY='取得したキー'
git push
```

Windows PowerShell の場合:

```powershell
$env:TYPESAFE_API_KEY = '取得したキー'
git push
```

環境変数はそのターミナルで起動した Git に渡ります。IDE から push する場合は、IDE を起動する環境にも設定してください。`TYPESAFEAI_API_KEY` も利用できます。キーがなければフックは理由を表示してスキップします。

## 実行範囲と手動確認

通常の `git push` では、更新済みのリモートブランチならそのリモート先との差分、初回 push ならローカル `main` との共通祖先からの差分を確認します。`review` はチェックアウト中の作業ツリーを読むため、別のブランチを指定して push した場合はその参照をスキップします。未コミットの変更があると読み取る内容に混ざり得るため、コミット後に実行してください。

API リクエスト前の件数と概算料金の確認:

```sh
npx -y jev-lint@0.7.0 review --base main --dry-run
```

手動で差分を確認する場合:

```sh
npx -y jev-lint@0.7.0 review --base main
```

クレジット切れ（API の HTTP 402）やサービス障害などで API リクエストに失敗したとき、`jev-lint` は理由を表示して終了コード 3 を返します。フックも API を利用できない旨を表示し、残りの Jev 確認を省いて push を続けます。ほかの Jev 実行エラーや指摘でも push は止めません。フックを使わない場合はローカル設定を解除します。

```sh
git config --local --unset core.hooksPath
```
