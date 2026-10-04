//! The same bundled OTF fonts are used on every platform.

use crate::settings::{AppSettings, UiLanguage};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use std::sync::Arc;

const JAPANESE_FONT_NAME: &str = "LINESeedJP_A_OTF_Rg";
const KOREAN_FONT_NAME: &str = "LINESeedKR-Rg";
const CHINESE_FONT_NAME: &str = "NotoSansCJKsc-VF";
const BUNDLED_FONTS: [(&str, &[u8]); 3] = [
    (
        JAPANESE_FONT_NAME,
        include_bytes!("../../../../assets/fonts/LINESeedJP_A_OTF_Rg.otf"),
    ),
    (
        KOREAN_FONT_NAME,
        include_bytes!("../../../../assets/fonts/LINESeedKR-Rg.otf"),
    ),
    (
        CHINESE_FONT_NAME,
        include_bytes!("../../../../assets/fonts/NotoSansCJKsc-VF.otf"),
    ),
];

#[derive(Resource, Default)]
pub(crate) struct UiFonts {
    applied_language: Option<UiLanguage>,
}

fn load_fonts(language: UiLanguage) -> egui::FontDefinitions {
    let mut definitions = egui::FontDefinitions::empty();
    for (name, bytes) in BUNDLED_FONTS {
        let mut data = egui::FontData::from_static(bytes);
        if name == CHINESE_FONT_NAME {
            // This variable OTF defaults to Thin (100); use Regular like
            // the two LINE Seed fonts.
            data.tweak.coords.push(b"wght", 400.0);
        }
        definitions
            .font_data
            .insert(name.to_owned(), Arc::new(data));
    }
    // Prefer the active language's glyph forms. Keep all three fonts available
    // for language buttons and model names, with LINE Seed KR before Noto's
    // broader CJK coverage so Korean text uses the bundled Korean typeface.
    let order = match language {
        UiLanguage::Ja | UiLanguage::En => {
            [JAPANESE_FONT_NAME, KOREAN_FONT_NAME, CHINESE_FONT_NAME]
        }
        UiLanguage::Ko => [KOREAN_FONT_NAME, JAPANESE_FONT_NAME, CHINESE_FONT_NAME],
        UiLanguage::Zh => [CHINESE_FONT_NAME, JAPANESE_FONT_NAME, KOREAN_FONT_NAME],
    };
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        definitions
            .families
            .insert(family, order.into_iter().map(str::to_owned).collect());
    }
    definitions
}

pub(crate) fn configure_fonts(
    mut contexts: EguiContexts,
    settings: Res<AppSettings>,
    mut state: ResMut<UiFonts>,
) -> Result {
    let language = settings.language();
    if state.applied_language == Some(language) {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    if state.applied_language.is_none() {
        let mut visuals = egui::Visuals::light();
        visuals.weak_text_color = Some(egui::Color32::from_gray(85));
        ctx.set_visuals(visuals);
        ctx.style_mut_of(ctx.theme(), |style| {
            style
                .text_styles
                .insert(egui::TextStyle::Small, egui::FontId::proportional(12.0));
        });
    }
    ctx.set_fonts(load_fonts(language));
    state.applied_language = Some(language);
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn bundled_otf_fonts_render_all_languages_after_switching() {
        let ctx = egui::Context::default();
        for language in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
            UiLanguage::Ja,
        ] {
            ctx.set_fonts(load_fonts(language));
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                for font in [
                    egui::FontId::proportional(16.0),
                    egui::FontId::monospace(16.0),
                ] {
                    let text = "日本語 ENGLISH 中文 한국어 VRM・カメラ選択 摄像头设置 카메라 설정";
                    ui.fonts_mut(|fonts| {
                        // Compare actual glyphs: egui 0.35's has_glyph compares
                        // font faces and rejects valid glyphs in the face that
                        // also provides the replacement character.
                        let glyph_uv = |fonts: &mut egui::epaint::text::FontsView<'_>,
                                        character: char| {
                            let galley = fonts.layout_no_wrap(
                                character.to_string(),
                                font.clone(),
                                egui::Color32::BLACK,
                            );
                            galley
                                .rows
                                .first()
                                .and_then(|row| row.glyphs.first())
                                .expect("one visible character produces a glyph")
                                .uv_rect
                        };
                        let replacement = glyph_uv(fonts, '\u{10ffff}');
                        for character in text.chars().filter(|c| !c.is_whitespace()) {
                            let uv = glyph_uv(fonts, character);
                            assert!(!uv.is_nothing(), "{language:?}: {character:?}");
                            assert_ne!(uv, replacement, "{language:?}: {character:?}");
                        }
                    });
                }
            });
        }
    }
}
