#Requires -Version 7.0
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'website'
$output = Join-Path $root 'target/website'
New-Item -ItemType Directory -Force -Path "$output/assets", "$output/manual" | Out-Null
Copy-Item -LiteralPath "$source/index.html" -Destination "$output/index.html"
Copy-Item -LiteralPath "$source/assets/site.css" -Destination "$output/assets/site.css"
Copy-Item -LiteralPath "$root/assets/icons/rustuberv-app-icon.svg" -Destination "$output/assets/app-icon.svg"
Copy-Item -LiteralPath "$root/assets/icons/rustuberv-app-icon-256.png" -Destination "$output/assets/app-icon.png"
Set-Content -LiteralPath "$output/.nojekyll" -Value '' -Encoding utf8NoBOM

$languages = @(
    @{ Code = 'ja'; Name = '日本語'; Title = '利用マニュアル'; Product = '製品紹介'; Guide = 'マニュアル'; Download = 'ダウンロード'; Toc = 'このページの内容'; Source = 'GitHubで読む'; Skip = '本文へ移動'; Top = 'ページの先頭へ'; Help = 'お問い合わせ・不具合報告'; Nav = 'メインナビゲーション'; Description = 'RusTuberVの画像付き利用マニュアル。VRMの読み込み、Webカメラの設定、表情と画角の調整、OBSへのNDI・Spout2出力、Looking Glassの使い方。' },
    @{ Code = 'en'; Name = 'English'; Title = 'User guide'; Product = 'Product'; Guide = 'User guide'; Download = 'Download'; Toc = 'On this page'; Source = 'Read on GitHub'; Skip = 'Skip to content'; Top = 'Back to top'; Help = 'Help & issues'; Nav = 'Main navigation'; Description = 'The illustrated RusTuberV user guide: load a VRM avatar, set up a webcam, adjust expressions and framing, send video to OBS with NDI or Spout2, and use Looking Glass.' },
    @{ Code = 'zh-CN'; Name = '简体中文'; Title = '使用手册'; Product = '产品介绍'; Guide = '使用手册'; Download = '下载'; Toc = '本页内容'; Source = '在GitHub上阅读'; Skip = '跳到正文'; Top = '返回顶部'; Help = '咨询与问题反馈'; Nav = '主导航'; Description = 'RusTuberV图文使用手册：加载VRM虚拟形象、设置摄像头、调整表情和构图、通过NDI或Spout2向OBS输出视频，以及使用Looking Glass。' },
    @{ Code = 'ko'; Name = '한국어'; Title = '사용 설명서'; Product = '제품 소개'; Guide = '사용 설명서'; Download = '다운로드'; Toc = '이 페이지의 내용'; Source = 'GitHub에서 읽기'; Skip = '본문으로 이동'; Top = '맨 위로'; Help = '문의 및 문제 신고'; Nav = '주 탐색'; Description = 'RusTuberV 그림 설명서: VRM 아바타 불러오기, 웹캠 설정, 표정과 구도 조정, NDI 또는 Spout2로 OBS에 영상 보내기, Looking Glass 사용 방법.' }
)
$template = Get-Content -LiteralPath "$source/manual.html" -Raw
$sourceRef = if ($env:GITHUB_SHA) { $env:GITHUB_SHA } else { 'main' }

function ConvertTo-GuideHtml {
    param([string]$Markdown)

    $html = (ConvertFrom-Markdown -InputObject $Markdown).Html
    # Preserve GitHub-style Unicode anchors used by the existing manuals.
    $usedIds = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $html = [regex]::Replace($html, '<h([1-6])(?:\s+[^>]*)?>(.*?)</h\1>', {
        param($match)
        $text = [System.Net.WebUtility]::HtmlDecode(($match.Groups[2].Value -replace '<[^>]+>', ''))
        $slug = ($text.ToLowerInvariant() -replace '[^\p{L}\p{M}\p{N}_\-\s]', '') -replace '\s', '-'
        $id = $slug
        $suffix = 1
        while (-not $usedIds.Add($id)) { $id = "$slug-$suffix"; $suffix++ }
        '<h{0} id="{1}">{2}</h{0}>' -f $match.Groups[1].Value, $id, $match.Groups[2].Value
    })
    $html = [regex]::Replace($html, 'href="([^"#?]+)\.md([#?][^"]*)?"', {
        param($match)
        $path = $match.Groups[1].Value
        if ($path -match '^https?://') { return $match.Value }
        if ($path -match '^../../README\.(en|ko|zh-CN)$') {
            return 'href="https://github.com/matsuvr/RusTuberV/blob/main/{0}.md{1}"' -f ($path -replace '^../../', ''), $match.Groups[2].Value
        }
        $path = if ($path -match '(^|/)README$') { $path -replace 'README$', '' } else { "$path.html" }
        'href="{0}{1}"' -f $path, $match.Groups[2].Value
    })
    $html = $html -replace '<table>', '<div class="table-scroll" tabindex="0"><table>' -replace '</table>', '</table></div>'
    # The product hero is eager; long manual pages load screenshots as needed.
    $html = $html -replace '<img ', '<img loading="lazy" decoding="async" '
    return $html
}

function Write-GuidePage {
    param([string]$MarkdownPath, [string]$Destination, [hashtable]$Language, [string]$RelativeRoot, [string]$Title)

    $html = ConvertTo-GuideHtml (Get-Content -LiteralPath (Join-Path $root $MarkdownPath) -Raw)
    $toc = foreach ($heading in [regex]::Matches($html, '<h2 id="([^"]+)">(.*?)</h2>')) {
        '<a href="#{0}">{1}</a>' -f $heading.Groups[1].Value, $heading.Groups[2].Value
    }
    $languageLinks = foreach ($entry in $languages) {
        $current = if ($entry.Code -eq $Language.Code -and $MarkdownPath -ne 'manual/SPOUT2.md') { ' aria-current="page"' } else { '' }
        '<a href="{0}manual/{1}/" lang="{1}"{2}>{3}</a>' -f $RelativeRoot, $entry.Code, $current, $entry.Name
    }
    $values = @{
        LANG = $Language.Code; ROOT = $RelativeRoot; TITLE = $Title; DESCRIPTION = $Language.Description
        PRODUCT_LABEL = $Language.Product; GUIDE_LABEL = $Language.Guide; DOWNLOAD_LABEL = $Language.Download
        TOC_LABEL = $Language.Toc; SOURCE_LABEL = $Language.Source; SKIP_LABEL = $Language.Skip
        TOP_LABEL = $Language.Top; HELP_LABEL = $Language.Help; NAV_LABEL = $Language.Nav
        GUIDE_URL = "${RelativeRoot}manual/$($Language.Code)/"
        SOURCE_URL = "https://github.com/matsuvr/RusTuberV/blob/$sourceRef/$MarkdownPath"
    }
    $page = $template
    foreach ($key in $values.Keys) { $page = $page.Replace("{{$key}}", [System.Net.WebUtility]::HtmlEncode($values[$key])) }
    $page = $page.Replace('{{TOC}}', ($toc -join "`n")).Replace('{{LANGUAGES}}', ($languageLinks -join "`n")).Replace('{{CONTENT}}', $html)
    Set-Content -LiteralPath (Join-Path $output $Destination) -Value $page -Encoding utf8NoBOM
}

foreach ($language in $languages) {
    $code = $language.Code
    New-Item -ItemType Directory -Force -Path "$output/manual/$code/images" | Out-Null
    Copy-Item -Path "$root/manual/$code/images/*.png" -Destination "$output/manual/$code/images"
    Write-GuidePage -MarkdownPath "manual/$code/README.md" -Destination "manual/$code/index.html" -Language $language -RelativeRoot '../../' -Title $language.Title
}
Write-GuidePage -MarkdownPath 'manual/SPOUT2.md' -Destination 'manual/SPOUT2.html' -Language $languages[0] -RelativeRoot '../' -Title 'Spout2でOBSに映す（Windows）'
Write-Output "Website built: $output"
