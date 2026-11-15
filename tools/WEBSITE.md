# GitHub Pages

製品紹介と利用マニュアルの公開先は <https://matsuvr.github.io/RusTuberV/> です。

## 編集とプレビュー

- 製品紹介：`website/index.html`
- 共通デザイン：`website/assets/site.css`
- マニュアルのレイアウト：`website/manual.html`
- マニュアルの本文・画像：`manual/`（配布ZIPと共通）

PowerShell 7で、リポジトリのルートから実行します。追加パッケージは不要です。

```powershell
.\tools\build-website.ps1
python -m http.server 4173 --bind 127.0.0.1 --directory target/website
```

`http://127.0.0.1:4173/` で確認できます。生成先の `target/website/` はコミットしません。

## 公開

GitHubのSettings → Pages → Sourceで **GitHub Actions** を選びます。
`main` またはサイト用の `docs/github-pages` ブランチで、サイト・マニュアル・アイコン・生成スクリプトを更新すると、`.github/workflows/pages.yml` が生成して公開します。
手動実行も同じワークフローを使います。`github-pages` environmentにブランチ制限がある場合は、公開するブランチを許可します。

マニュアルはMarkdownからHTMLに変換し、相対リンクとUnicodeの見出しアンカーを引き継ぎます。GitHubで本文を読むリンクは、ビルド元のコミットを参照します。
