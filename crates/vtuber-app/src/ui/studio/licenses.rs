//! Bundled asset and dependency license display.
use super::section;
use crate::licenses::LicenseGroup;
use crate::settings::UiLanguage;
use bevy_egui::egui::{self, RichText, Ui};

/// Licenses of the built application. Only the headings are localized; every
/// license body stays in the language its author published it in.
pub(super) fn oss_licenses_page(ui: &mut Ui, lang: UiLanguage) {
    section(
        ui,
        lang.pick("同梱アセット", "Bundled assets", "内置资源", "번들 에셋"),
        |ui| {
            license_groups(ui, "bundled", crate::licenses::bundled_groups(), lang);
        },
    );
    section(
        ui,
        lang.pick(
            "依存パッケージ",
            "Dependency packages",
            "依赖包",
            "의존 패키지",
        ),
        |ui| {
            license_groups(
                ui,
                "dependencies",
                crate::licenses::dependency_groups(),
                lang,
            );
        },
    );
}

fn license_groups(ui: &mut Ui, list_id: &str, groups: &[LicenseGroup], lang: UiLanguage) {
    for (group_index, group) in groups.iter().enumerate() {
        let heading = format!(
            "{}  ({} {})",
            group.expression,
            group.items.len(),
            lang.pick("件", "items", "项", "개")
        );
        egui::CollapsingHeader::new(RichText::new(heading).strong())
            .id_salt(("oss_licenses", list_id, group_index))
            .show(ui, |ui| {
                let mut items = group
                    .items
                    .iter()
                    .map(|item| {
                        if item.version.is_empty() {
                            item.name.to_string()
                        } else {
                            format!("{} {}", item.name, item.version)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                items.push('\n');
                ui.label(RichText::new(items).monospace());
                if group.texts.is_empty() {
                    ui.label(lang.pick(
                        "このパッケージはライセンス本文を同梱していません。上流の配布元を参照してください。",
                        "These packages ship no license text. See their upstream distribution.",
                        "这些包未附带许可证正文，请参阅其上游分发。",
                        "이 패키지에는 라이선스 본문이 포함되어 있지 않습니다. 업스트림 배포를 참조하세요.",
                    ));
                }
                for (text_index, text) in group.texts.iter().enumerate() {
                    // Two crates under one expression rarely share a file, and
                    // their copyright lines are what tells them apart.
                    let holder = text.body.lines().next().unwrap_or_default().trim();
                    let title = if holder.is_empty() {
                        text.file_name.to_string()
                    } else {
                        format!("{} — {holder}", text.file_name)
                    };
                    egui::CollapsingHeader::new(RichText::new(title).monospace())
                        .id_salt(("license_text", text_index))
                        .show(ui, |ui| {
                            ui.label(RichText::new(text.body).monospace());
                        });
                }
            });
    }
}
