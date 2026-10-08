//! VRM import license confirmation.
use super::{RichText, Ui, UiLanguage, UiState, UiViewModel, egui, primary_button};
use crate::{actions::UiAction, license_review::VrmLicenseReview};
use bevy_egui::egui::{Color32, Id, vec2};

/// Renders the license review sheet on top of the workspace when a model is
/// waiting for acceptance. Returns whether the pointer is over any UI.
pub(super) fn render_avatar_import_review(
    ctx: &egui::Context,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
    over_ui: bool,
) -> bool {
    let Some(review) = &vm.avatar_import_review.review else {
        return over_ui;
    };
    avatar_import_review_modal(ctx, review, vm.avatar_import_review.accepted, state, lang);
    true
}

/// Apple-style centered consent sheet shown before an imported VRM reaches the
/// asset store. The newly selected model stays unloaded until this sheet is
/// accepted.
fn avatar_import_review_modal(
    ctx: &egui::Context,
    review: &VrmLicenseReview,
    accepted: bool,
    state: &mut UiState,
    lang: UiLanguage,
) {
    let mut checked = accepted;
    let mut import_requested = false;
    let mut cancel_requested = false;
    let viewport = ctx.content_rect();
    let frame = egui::Frame::popup(&ctx.global_style());
    let content_width = viewport.width() * 0.8 - frame.total_margin().sum().x;
    let modal = egui::Modal::new(Id::new("avatar_import_review"))
        .frame(frame)
        .show(ctx, |ui| {
            ui.set_width(content_width);
            // Horizontal metadata rows must wrap too, including unbroken URLs.
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            egui::ScrollArea::vertical()
                .id_salt("avatar_import_review_details")
                .max_height(viewport.height() * 0.6)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    review_sheet_header(ui, review, lang);
                    ui.add_space(12.0);
                    ui.separator();
                    review_sheet_details(ui, review, lang);
                });
            ui.separator();
            ui.add_space(8.0);
            ui.checkbox(
                &mut checked,
                lang.pick(
                    "この VRM のライセンス・利用条件を確認しました",
                    "I have reviewed this VRM's license and usage terms",
                    "我已确认该 VRM 的许可与使用条件",
                    "이 VRM의 라이선스 및 이용 조건을 확인했습니다",
                ),
            );
            ui.add_space(8.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if primary_button(
                    ui,
                    lang.pick("読み込む", "Import", "导入", "불러오기"),
                    checked,
                )
                .clicked()
                {
                    import_requested = true;
                }
                if ui
                    .button(lang.pick("キャンセル", "Cancel", "取消", "취소"))
                    .clicked()
                {
                    cancel_requested = true;
                }
            });
        });
    if checked != accepted {
        state.emit(UiAction::SetAvatarImportReviewAccepted { accepted: checked });
    }
    if modal.should_close() || cancel_requested {
        state.emit(UiAction::CancelAvatarImportReview);
    } else if import_requested {
        state.emit(UiAction::AcceptAvatarImportReview);
    }
}

fn review_sheet_header(ui: &mut Ui, review: &VrmLicenseReview, lang: UiLanguage) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(44.0, 44.0), egui::Sense::hover());
        ui.painter()
            .circle_filled(rect.center(), 22.0, Color32::from_gray(238));
        paint_license_glyph(ui.painter(), rect.center(), Color32::from_gray(80));
        ui.add_space(10.0);
        ui.vertical(|ui| {
            ui.label(
                RichText::new(lang.pick(
                    "ライセンス確認",
                    "License review",
                    "许可确认",
                    "라이선스 확인",
                ))
                .size(19.0)
                .strong(),
            );
            ui.label(RichText::new(&review.model_name).size(15.0));
            ui.label(
                RichText::new(match review.generation {
                    crate::import::VrmGeneration::Vrm0 => "VRM 0.x",
                    crate::import::VrmGeneration::Vrm1 => "VRM 1.0",
                })
                .small()
                .weak(),
            );
        });
    });
}

fn review_sheet_details(ui: &mut Ui, review: &VrmLicenseReview, lang: UiLanguage) {
    review_sheet_group(
        ui,
        lang.pick("モデル情報", "Model information", "模型信息", "모델 정보"),
    );
    review_field(
        ui,
        lang.pick("バージョン", "Version", "版本", "버전"),
        review.version.as_deref(),
        lang,
    );
    let authors = joined_values(&review.authors);
    review_field(
        ui,
        lang.pick("作者", "Author", "作者", "제작자"),
        authors.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("連絡先", "Contact", "联系方式", "연락처"),
        review.contact_information.as_deref(),
        lang,
    );
    let references = joined_values(&review.references);
    review_field(
        ui,
        lang.pick("参照", "Reference", "参考", "참조"),
        references.as_deref(),
        lang,
    );

    review_sheet_group(
        ui,
        lang.pick("利用条件", "Usage permission", "使用条件", "이용 조건"),
    );
    review_field(
        ui,
        lang.pick(
            "アバター利用",
            "Avatar permission",
            "虚拟形象使用",
            "아바타 이용",
        ),
        review.avatar_permission.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("暴力表現", "Violent", "暴力表现", "폭력 표현"),
        review.allow_violent_usage.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("性的表現", "Sexual", "性表现", "성적 표현"),
        review.allow_sexual_usage.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("商用利用", "Commercial", "商业用途", "상업적 이용"),
        review.commercial_usage.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("その他の許諾", "Other permission", "其他许可", "기타 허가"),
        review.other_permission_url.as_deref(),
        lang,
    );

    review_sheet_group(
        ui,
        lang.pick(
            "配布・改変ライセンス",
            "Distribution & modification",
            "分发与修改许可",
            "배포·개조 라이선스",
        ),
    );
    review_field(
        ui,
        lang.pick("改変", "Modification", "修改", "개조"),
        review.modification_license.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("ライセンス名", "License name", "许可名称", "라이선스 이름"),
        review.license_name.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("ライセンスURL", "License URL", "许可URL", "라이선스 URL"),
        review.license_url.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick(
            "その他ライセンスURL",
            "Other license URL",
            "其他许可URL",
            "기타 라이선스 URL",
        ),
        review.other_license_url.as_deref(),
        lang,
    );
}

fn review_sheet_group(ui: &mut Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title).size(14.0).strong());
    ui.add_space(4.0);
}

fn review_field(ui: &mut Ui, label: &str, value: Option<&str>, lang: UiLanguage) {
    let label_width = (ui.available_width() * 0.3).min(150.0);
    ui.horizontal_top(|ui| {
        ui.add_sized(
            [label_width, 18.0],
            egui::Label::new(RichText::new(label).weak()).wrap(),
        );
        match value.filter(|value| !value.is_empty()) {
            Some(value) if is_external_url(value) => {
                ui.hyperlink_to(value, value);
            }
            Some(value) => {
                ui.label(value);
            }
            None => {
                ui.label(
                    RichText::new(lang.pick("未指定", "Not specified", "未指定", "미지정")).weak(),
                );
            }
        }
    });
}

fn joined_values(values: &[String]) -> Option<String> {
    (!values.is_empty()).then(|| values.join(", "))
}

fn is_external_url(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("http://")
}

/// Minimal shield-and-check glyph for the license consent sheet, drawn in the
/// same monochrome style as the other controls.
fn paint_license_glyph(painter: &egui::Painter, center: egui::Pos2, color: Color32) {
    let stroke = egui::Stroke::new(1.6, color);
    let width = 11.0;
    let top = center.y - 12.5;
    let shoulder = center.y + 1.0;
    painter.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(center.x, top),
            egui::pos2(center.x + width, top + 4.0),
            egui::pos2(center.x + width, shoulder),
            egui::pos2(center.x, top + 24.0),
            egui::pos2(center.x - width, shoulder),
            egui::pos2(center.x - width, top + 4.0),
        ],
        Color32::TRANSPARENT,
        stroke,
    ));
    painter.line_segment(
        [
            egui::pos2(center.x - 5.0, center.y - 1.0),
            egui::pos2(center.x - 1.5, center.y + 2.5),
        ],
        stroke,
    );
    painter.line_segment(
        [
            egui::pos2(center.x - 1.5, center.y + 2.5),
            egui::pos2(center.x + 5.5, center.y - 4.5),
        ],
        stroke,
    );
}

#[cfg(test)]
pub(super) fn review_fixture() -> VrmLicenseReview {
    VrmLicenseReview {
        generation: crate::import::VrmGeneration::Vrm1,
        model_name: "Test model".to_string(),
        version: None,
        authors: vec!["Author".to_string()],
        contact_information: None,
        references: Vec::new(),
        avatar_permission: Some("everyone".to_string()),
        allow_violent_usage: Some("false".to_string()),
        allow_sexual_usage: Some("false".to_string()),
        commercial_usage: Some("personalNonProfit".to_string()),
        other_permission_url: None,
        modification_license: Some("prohibited".to_string()),
        license_name: None,
        license_url: None,
        other_license_url: None,
        source_path: std::path::PathBuf::from("model.vrm"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn license_review_modal_escape_emits_cancel() {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let review = review_fixture();
        // Frame 1 registers the modal; frame 2 receives the Escape.
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            avatar_import_review_modal(ui.ctx(), &review, false, &mut state, UiLanguage::Ja);
        });
        let mut input = egui::RawInput::default();
        input.events.push(egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run_ui(input, |ui| {
            avatar_import_review_modal(ui.ctx(), &review, false, &mut state, UiLanguage::Ja);
        });
        assert!(
            state
                .pending_actions
                .contains(&UiAction::CancelAvatarImportReview)
        );
    }

    #[test]
    fn license_review_modal_does_not_import_while_unchecked() {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let review = review_fixture();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            avatar_import_review_modal(ui.ctx(), &review, false, &mut state, UiLanguage::Ja);
        });
        assert!(state.pending_actions.is_empty());
    }
}
