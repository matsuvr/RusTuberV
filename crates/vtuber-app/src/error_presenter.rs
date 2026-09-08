//! Maps domain errors to localized user-facing summaries.
//! Technical backend details retain their original text.

use bevy::prelude::*;
use crate::actions::UiAction;
use crate::settings::UiLanguage;

/// A user-facing error presentation.
#[derive(Clone, Debug, PartialEq)]
pub struct ErrorPresentation {
    /// Stable technical error code.
    pub code: &'static str,
    /// Localized summary.
    pub user_message: String,
    /// Suggested recovery actions.
    pub suggested_actions: Vec<UiAction>,
}

/// Present a domain error in one of the four supported languages.
#[must_use]
pub fn present_error(error: &crate::orchestrator::OrchestratorError, language: UiLanguage) -> ErrorPresentation {
    use crate::orchestrator::OrchestratorError;
    let message = |ja: &'static str, en: &'static str, zh: &'static str, ko: &'static str| language.pick(ja, en, zh, ko).to_string();
    match error {
        OrchestratorError::ImportFailed(detail) => ErrorPresentation {
            code: "IMPORT_FAILED",
            user_message: format!("{}: {detail}", language.pick("モデルを読み込めませんでした", "Could not import model", "无法加载模型", "모델을 불러오지 못했습니다")),
            suggested_actions: vec![UiAction::DismissError],
        },
        OrchestratorError::NoCameraSelected => ErrorPresentation {
            code: "NO_CAMERA",
            user_message: message("開始する前にカメラを選択してください。", "Please select a camera before starting.", "开始前请选择摄像头。", "시작하기 전에 카메라를 선택하세요."),
            suggested_actions: vec![UiAction::RefreshCameras, UiAction::DismissError],
        },
        OrchestratorError::NoAvatarLoaded => ErrorPresentation {
            code: "NO_AVATAR",
            user_message: message("開始する前にアバターを読み込んでください。", "Please import an avatar before starting.", "开始前请加载虚拟形象。", "시작하기 전에 아바타를 불러오세요."),
            suggested_actions: vec![UiAction::DismissError],
        },
        OrchestratorError::PipelineAlreadyRunning => ErrorPresentation {
            code: "PIPELINE_RUNNING",
            user_message: message("トラッキングは既に実行中です。", "Tracking is already running.", "跟踪已在运行。", "트래킹이 이미 실행 중입니다."),
            suggested_actions: vec![UiAction::DismissError],
        },
        OrchestratorError::PipelineNotRunning => ErrorPresentation {
            code: "PIPELINE_NOT_RUNNING",
            user_message: message("トラッキングは実行されていません。", "Tracking is not running.", "跟踪未在运行。", "트래킹이 실행 중이 아닙니다."),
            suggested_actions: vec![UiAction::DismissError],
        },
        OrchestratorError::AvatarLoadRejected(_) => ErrorPresentation {
            code: "AVATAR_LOAD_REJECTED",
            user_message: message("アバターを読み込めませんでした。モデルを確認して再試行してください。", "The avatar could not be loaded. Check the model and try again.", "无法加载虚拟形象。请检查模型后重试。", "아바타를 불러오지 못했습니다. 모델을 확인하고 다시 시도하세요."),
            suggested_actions: vec![UiAction::RetryAfterError, UiAction::DismissError],
        },
        OrchestratorError::AvatarLifecycleFailed(_) => ErrorPresentation {
            code: "AVATAR_LIFECYCLE_FAILED",
            user_message: message("アバターの読み込みまたはバインド中に失敗しました。再試行してください。", "The avatar failed during loading or binding. Try again.", "虚拟形象加载或绑定失败，请重试。", "아바타를 불러오거나 바인딩하는 중 실패했습니다. 다시 시도하세요."),
            suggested_actions: vec![UiAction::RetryAfterError, UiAction::DismissError],
        },
        OrchestratorError::ArmPoseSettingsFailed(_) => ErrorPresentation {
            code: "ARM_POSE_SETTINGS_FAILED",
            user_message: message("設定を保存できませんでした。", "The setting could not be saved.", "无法保存设置。", "설정을 저장하지 못했습니다."),
            suggested_actions: vec![UiAction::DismissError],
        },
        OrchestratorError::CameraFailed(_) => ErrorPresentation {
            code: "CAMERA_FAILED",
            user_message: message("カメラを開始できなかったか、切断されました。", "The camera could not be started or was disconnected.", "无法启动摄像头，或摄像头已断开连接。", "카메라를 시작하지 못했거나 연결이 끊겼습니다."),
            suggested_actions: vec![UiAction::RefreshCameras, UiAction::DismissError],
        },
        OrchestratorError::InferenceFailed(_) => ErrorPresentation {
            code: "INFERENCE_FAILED",
            user_message: message("顔トラッキングモデルの読み込みまたは実行に失敗しました。", "The face tracking model failed to load or run.", "面部跟踪模型加载或运行失败。", "얼굴 트래킹 모델을 불러오거나 실행하는 데 실패했습니다."),
            suggested_actions: vec![UiAction::RetryAfterError, UiAction::DismissError],
        },
    }
}

/// Tracks the currently presented error, including language changes.
#[derive(Resource, Debug, Default)]
pub struct ErrorPresenter {
    last_presented_code: Option<String>,
    current: Option<ErrorPresentation>,
}
impl ErrorPresenter {
    /// Returns true when a different presentation replaces the current one.
    pub fn update(&mut self, error: Option<&crate::orchestrator::OrchestratorError>, language: UiLanguage) -> bool {
        match error {
            Some(error) => {
                let presentation = present_error(error, language);
                if self.current.as_ref() == Some(&presentation) { return false; }
                self.last_presented_code = Some(presentation.code.to_owned());
                self.current = Some(presentation);
                true
            }
            None => { self.last_presented_code = None; self.current = None; false }
        }
    }
    /// Current presentation.
    #[must_use]
    pub fn current(&self) -> Option<&ErrorPresentation> { self.current.as_ref() }
    /// Dismiss the presentation.
    pub fn dismiss(&mut self) { self.current = None; self.last_presented_code = None; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::OrchestratorError;
    #[test]
    fn error_presenter_no_error() {
        let mut presenter = ErrorPresenter::default(); assert!(!presenter.update(None, UiLanguage::Ja)); assert!(presenter.current().is_none());
    }
    #[test]
    fn error_presenter_new_error() {
        let mut presenter = ErrorPresenter::default();
        assert!(presenter.update(Some(&OrchestratorError::NoCameraSelected), UiLanguage::Ja));
        assert_eq!(presenter.current().unwrap().code, "NO_CAMERA");
    }
    #[test]
    fn error_presenter_same_error_not_re_presented() {
        let mut presenter = ErrorPresenter::default(); let error = OrchestratorError::NoCameraSelected;
        assert!(presenter.update(Some(&error), UiLanguage::Ja)); assert!(!presenter.update(Some(&error), UiLanguage::Ja));
    }
    #[test]
    fn error_presenter_language_switch_re_presents() {
        let mut presenter = ErrorPresenter::default(); let error = OrchestratorError::NoCameraSelected;
        for language in [UiLanguage::Ja, UiLanguage::En, UiLanguage::Zh, UiLanguage::Ko] {
            assert!(presenter.update(Some(&error), language));
            assert!(!presenter.update(Some(&error), language));
            assert!(!presenter.current().unwrap().user_message.is_empty());
        }
    }
    #[test]
    fn error_presenter_different_error_re_presented() {
        let mut presenter = ErrorPresenter::default();
        assert!(presenter.update(Some(&OrchestratorError::NoCameraSelected), UiLanguage::Ja));
        assert!(presenter.update(Some(&OrchestratorError::NoAvatarLoaded), UiLanguage::Ja));
        assert_eq!(presenter.current().unwrap().code, "NO_AVATAR");
    }
    #[test]
    fn error_presenter_dismiss() {
        let mut presenter = ErrorPresenter::default(); presenter.update(Some(&OrchestratorError::NoCameraSelected), UiLanguage::Ja);
        presenter.dismiss(); assert!(presenter.current().is_none());
    }
    #[test]
    fn present_error_defaults_to_japanese_and_keeps_detail() {
        let presentation = present_error(&OrchestratorError::ImportFailed("bad file".to_owned()), UiLanguage::Ja);
        assert_eq!(presentation.code, "IMPORT_FAILED"); assert!(presentation.user_message.starts_with("モデルを読み込めませんでした"));
        assert!(presentation.user_message.contains("bad file")); assert!(presentation.suggested_actions.contains(&UiAction::DismissError));
    }
    #[test]
    fn present_error_no_camera_has_refresh() {
        let presentation = present_error(&OrchestratorError::NoCameraSelected, UiLanguage::En);
        assert!(presentation.suggested_actions.contains(&UiAction::RefreshCameras));
        assert!(presentation.user_message.starts_with("Please select a camera"));
    }
}
