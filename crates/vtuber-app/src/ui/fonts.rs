//! Font I/O is separate from UI rendering. No system font is copied or distributed.

use crate::settings::{AppSettings, UiLanguage};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use std::{io, path::PathBuf, sync::Arc};

const JAPANESE_FONT_NAME: &str = "LINESeedJP_A_Rg";
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

fn font_definitions(language: UiLanguage) -> io::Result<egui::FontDefinitions> {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        JAPANESE_FONT_NAME.to_owned(),
        Arc::new(egui::FontData::from_static(JAPANESE_FONT_BYTES)),
    );
    let locale_font = match language {
        UiLanguage::Ja | UiLanguage::En => None,
        UiLanguage::Zh | UiLanguage::Ko => {
            let path = system_font_path(language)?;
            let data = std::fs::read(&path).map_err(|error| {
                io::Error::new(error.kind(), format!("{}: {error}", path.display()))
            })?;
            fonts.font_data.insert(
                "locale-system-font".to_owned(),
                Arc::new(egui::FontData::from_owned(data)),
            );
            Some("locale-system-font")
        }
    };
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        let names = fonts.families.entry(family).or_default();
        names.insert(0, JAPANESE_FONT_NAME.to_owned());
        if let Some(name) = locale_font {
            names.insert(0, name.to_owned());
        }
    }
    Ok(fonts)
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
    match font_definitions(language) {
        Ok(fonts) => {
            ctx.set_fonts(fonts);
            state.error = None;
        }
        Err(error) => {
            // Do not silently switch languages or download fonts. The studio
            // displays this error with explicit Japanese/English recovery buttons.
            state.error = Some(error.to_string());
        }
    }
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

    #[test]
    fn japanese_and_english_need_no_system_font_or_network() {
        for language in [UiLanguage::Ja, UiLanguage::En] {
            let fonts = font_definitions(language).expect("bundled font");
            assert_eq!(
                fonts.font_data[JAPANESE_FONT_NAME].font.as_ref(),
                JAPANESE_FONT_BYTES
            );
        }
    }
}
