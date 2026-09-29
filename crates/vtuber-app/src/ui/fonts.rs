//! Font I/O is separate from UI rendering. No system font is copied or distributed.

use crate::settings::{AppSettings, UiLanguage};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use std::{io, path::PathBuf, sync::Arc};

const JAPANESE_FONT_NAME: &str = "LINESeedJP_A_Rg";
const CHINESE_FONT_NAME: &str = "zh-system-font";
const KOREAN_FONT_NAME: &str = "ko-system-font";
static JAPANESE_FONT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/fonts/LINESeedJP_A_TTF_Rg.ttf"
));

#[derive(Resource, Default)]
pub(crate) struct UiFonts {
    attempted_language: Option<UiLanguage>,
    pub error: Option<String>,
}

fn system_font_path(language: UiLanguage) -> io::Result<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        let windows = std::env::var_os("WINDIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "WINDIR is not set"))?;
        let filename = match language {
            UiLanguage::Zh => "msyh.ttc",
            UiLanguage::Ko => "malgun.ttf",
            UiLanguage::Ja | UiLanguage::En => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Japanese and English use the bundled font",
                ));
            }
        };
        Ok(PathBuf::from(windows).join("Fonts").join(filename))
    }
    #[cfg(target_os = "macos")]
    {
        let filename = match language {
            UiLanguage::Zh => "PingFang.ttc",
            UiLanguage::Ko => "AppleSDGothicNeo.ttc",
            UiLanguage::Ja | UiLanguage::En => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Japanese and English use the bundled font",
                ));
            }
        };
        Ok(PathBuf::from("/System/Library/Fonts").join(filename))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let _ = language;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "CJK system font loading supports Windows and macOS",
        ))
    }
}

/// The CJK system fonts the four language buttons need, in fallback order
/// after the bundled font.
const CJK_SYSTEM_FONTS: [(UiLanguage, &str); 2] = [
    (UiLanguage::Zh, CHINESE_FONT_NAME),
    (UiLanguage::Ko, KOREAN_FONT_NAME),
];

/// What the egui context gets, plus the system fonts this machine could not
/// provide. Both are reported to the user; neither is allowed to take the
/// bundled font down with it.
struct LoadedFonts {
    definitions: egui::FontDefinitions,
    unavailable: Vec<String>,
}

fn read_system_font(language: UiLanguage) -> io::Result<Vec<u8>> {
    let path = system_font_path(language)?;
    std::fs::read(&path)
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", path.display())))
}

fn load_fonts(language: UiLanguage) -> LoadedFonts {
    load_fonts_with(language, read_system_font)
}

fn load_fonts_with(
    language: UiLanguage,
    read: impl Fn(UiLanguage) -> io::Result<Vec<u8>>,
) -> LoadedFonts {
    let mut definitions = egui::FontDefinitions::default();
    definitions.font_data.insert(
        JAPANESE_FONT_NAME.to_owned(),
        Arc::new(egui::FontData::from_static(JAPANESE_FONT_BYTES)),
    );
    // Every language button is visible in every language, so both CJK system
    // fonts are read regardless of the active language. One that this machine
    // does not provide is reported and left out of the families below.
    let mut loaded: Vec<&str> = Vec::new();
    let mut unavailable = Vec::new();
    for (font_language, name) in CJK_SYSTEM_FONTS {
        match read(font_language) {
            Ok(data) => {
                definitions
                    .font_data
                    .insert(name.to_owned(), Arc::new(egui::FontData::from_owned(data)));
                loaded.push(name);
            }
            Err(error) => unavailable.push(error.to_string()),
        }
    }
    let preferred = match language {
        UiLanguage::Zh => CHINESE_FONT_NAME,
        UiLanguage::Ko => KOREAN_FONT_NAME,
        UiLanguage::Ja | UiLanguage::En => JAPANESE_FONT_NAME,
    };
    let mut order = Vec::new();
    if loaded.contains(&preferred) {
        order.push(preferred);
    }
    order.push(JAPANESE_FONT_NAME);
    for name in loaded {
        if !order.contains(&name) {
            order.push(name);
        }
    }
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        let names = definitions.families.entry(family).or_default();
        for name in order.iter().rev() {
            names.insert(0, (*name).to_owned());
        }
    }
    LoadedFonts {
        definitions,
        unavailable,
    }
}

pub(crate) fn configure_fonts(
    mut contexts: EguiContexts,
    settings: Res<AppSettings>,
    mut state: ResMut<UiFonts>,
) -> Result {
    let language = settings.language();
    if state.attempted_language == Some(language) {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    if state.attempted_language.is_none() {
        ctx.set_visuals(egui::Visuals::light());
    }
    state.attempted_language = Some(language);
    // The bundled font is applied no matter what the system fonts did, so a
    // missing one costs only its own language. Do not silently switch languages
    // or download fonts; the studio reports the failure with explicit
    // Japanese/English recovery buttons.
    let fonts = load_fonts(language);
    ctx.set_fonts(fonts.definitions);
    state.error = (!fonts.unavailable.is_empty()).then(|| fonts.unavailable.join("\n"));
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;

    /// Stands in for the system font reader so a missing font can be exercised
    /// without touching the fonts of the machine running the test.
    fn stub_read(missing: Vec<UiLanguage>) -> impl Fn(UiLanguage) -> io::Result<Vec<u8>> {
        move |language| {
            if missing.contains(&language) {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("stubbed missing {language:?}"),
                ))
            } else {
                Ok(vec![0_u8; 4])
            }
        }
    }

    fn family_names(fonts: &egui::FontDefinitions) -> &Vec<String> {
        &fonts.families[&egui::FontFamily::Proportional]
    }

    #[test]
    fn both_cjk_system_fonts_render_every_language_button() {
        for language in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            let fonts = load_fonts_with(language, stub_read(Vec::new()));
            assert_eq!(
                fonts.definitions.font_data[JAPANESE_FONT_NAME]
                    .font
                    .as_ref(),
                JAPANESE_FONT_BYTES
            );
            for name in [CHINESE_FONT_NAME, KOREAN_FONT_NAME] {
                assert!(fonts.definitions.font_data.contains_key(name));
                assert!(family_names(&fonts.definitions).contains(&name.to_owned()));
            }
            assert!(fonts.unavailable.is_empty());
        }
    }

    /// The regression: one unreadable system font used to fail the whole
    /// definition set, so even the bundled Japanese font was never applied.
    #[test]
    fn a_missing_system_font_keeps_the_bundled_font_and_is_reported() {
        for language in [UiLanguage::Ja, UiLanguage::En] {
            let fonts = load_fonts_with(language, stub_read(vec![UiLanguage::Ko]));
            assert_eq!(
                fonts.definitions.font_data[JAPANESE_FONT_NAME]
                    .font
                    .as_ref(),
                JAPANESE_FONT_BYTES,
                "{language:?} still renders with the bundled font"
            );
            let names = family_names(&fonts.definitions);
            assert_eq!(names.first().map(String::as_str), Some(JAPANESE_FONT_NAME));
            assert!(names.contains(&CHINESE_FONT_NAME.to_owned()));
            assert!(!fonts.definitions.font_data.contains_key(KOREAN_FONT_NAME));
            assert!(
                !names.contains(&KOREAN_FONT_NAME.to_owned()),
                "a name without font data must not be registered in a family"
            );
            assert_eq!(fonts.unavailable.len(), 1);
        }
    }

    #[test]
    fn the_active_language_leads_when_its_system_font_is_readable() {
        let fonts = load_fonts_with(UiLanguage::Zh, stub_read(Vec::new()));
        assert_eq!(
            family_names(&fonts.definitions).first().map(String::as_str),
            Some(CHINESE_FONT_NAME)
        );
        let fonts = load_fonts_with(UiLanguage::Ko, stub_read(Vec::new()));
        assert_eq!(
            family_names(&fonts.definitions).first().map(String::as_str),
            Some(KOREAN_FONT_NAME)
        );
    }
}
