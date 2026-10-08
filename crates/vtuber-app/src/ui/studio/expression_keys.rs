//! Expression assignment, status and keyboard input.
use super::section;
use super::{RichText, Ui, UiLanguage, UiState, UiViewModel, egui};
use crate::{
    actions::UiAction,
    expression_keys::ExpressionKey,
    ui_model::{AvatarLifecycleState, ExpressionEntryViewModel},
};
use vtuber_avatar::{ExpressionAvailability, ExpressionKind};

pub(super) fn expression_keys_page(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
) {
    expression_settings_section(ui, vm, state, lang);
    expression_status_section(ui, vm, state, lang);
}

/// Translates standard presets and keeps the exact runtime ID visible.
fn expression_entry_name(entry: &ExpressionEntryViewModel, lang: UiLanguage) -> String {
    // Only expressions classified as standard presets get a translated label.
    // A custom expression that happens to be named `happy` keeps its author
    // string and is never displayed as the standard 喜 label.
    let standard = match entry.kind {
        ExpressionKind::EmotionalPreset | ExpressionKind::Neutral => match entry.id.as_str() {
            "happy" => Some(lang.pick("喜", "Happy", "喜悦", "기쁨")),
            "angry" => Some(lang.pick("怒", "Angry", "愤怒", "분노")),
            "sad" => Some(lang.pick("哀", "Sad", "悲伤", "슬픔")),
            "relaxed" => Some(lang.pick("楽", "Relaxed", "放松", "편안함")),
            "surprised" => Some(lang.pick("驚き", "Surprised", "惊讶", "놀람")),
            "neutral" => Some(lang.pick("通常", "Neutral", "自然", "중립")),
            _ => None,
        },
        ExpressionKind::Custom | ExpressionKind::Tracking => None,
    };
    let base = match standard {
        Some(label) => format!("{label} ({})", entry.id),
        None => entry.source_name.clone(),
    };
    if entry.kind == ExpressionKind::Tracking {
        format!(
            "{} [{}]",
            base,
            lang.pick("追跡用", "Tracking", "跟踪用", "트래킹용")
        )
    } else {
        base
    }
}

/// Describes availability as a disabled-candidate suffix.
fn expression_availability_suffix(
    availability: &ExpressionAvailability,
    lang: UiLanguage,
) -> String {
    match availability {
        ExpressionAvailability::Ready => String::new(),
        ExpressionAvailability::Empty => {
            format!(" — {}", lang.pick("未定義", "Empty", "空定义", "비어 있음"))
        }
        ExpressionAvailability::Unresolved { reason } => format!(
            " — {}: {reason}",
            lang.pick("利用できません", "Unavailable", "不可用", "사용할 수 없음")
        ),
        ExpressionAvailability::Unsupported { reason } => format!(
            " — {}: {reason}",
            lang.pick("利用できません", "Unavailable", "不可用", "사용할 수 없음")
        ),
    }
}

fn expression_settings_section(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
) {
    section(
        ui,
        lang.pick(
            "表情・キー割り当て",
            "Expressions & Key Bindings",
            "表情与按键绑定",
            "표정 및 키 할당",
        ),
        |ui| {
            ui.label(lang.pick(
                "ESCまたはスペースキーで標準の表情に戻ります",
                "Press Escape or Space to return to the standard expression.",
                "按 Esc 或空格键可恢复为标准表情。",
                "Esc 또는 스페이스 키를 누르면 표준 표정으로 돌아갑니다.",
            ));
            ui.add_space(6.0);
            if !vm.expression.has_catalog {
                let message = if vm.avatar.imported_model.is_some() {
                    lang.pick(
                        "選択できる表情がありません",
                        "No selectable expressions.",
                        "没有可选择的表情。",
                        "선택할 수 있는 표정이 없습니다.",
                    )
                } else {
                    lang.pick(
                        "モデルを読み込んでください",
                        "Load a model to continue.",
                        "请先加载模型。",
                        "먼저 모델을 불러오세요.",
                    )
                };
                ui.label(message);
                return;
            }
            // Every assignment action carries the exact model/generation this
            // snapshot was built for.
            let target = vm.expression.target.clone();
            ui.label(lang.pick(
                "最大36個のキーに割り当てられます。自動割り当てに含まれなかった表情も、ここで選択できます。",
                "Up to 36 keys can be assigned. You can also choose expressions that were not assigned automatically.",
                "最多可绑定36个按键。未被自动分配的表情也可以在这里选择。",
                "최대 36개 키에 할당할 수 있습니다. 자동 할당되지 않은 표정도 여기에서 선택할 수 있습니다.",
            ));
            ui.add_space(6.0);
            for row in &vm.expression.bindings {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [28.0, 18.0],
                        egui::Label::new(RichText::new(row.key.label()).strong()),
                    );
                    let selected_label = row
                        .expression
                        .as_deref()
                        .and_then(|id| {
                            vm.expression
                                .entries
                                .iter()
                                .find(|entry| entry.id == id)
                                .map(|entry| expression_entry_name(entry, lang))
                        })
                        .unwrap_or_else(|| {
                            row.expression.clone().unwrap_or_else(|| {
                                lang.pick("未割り当て", "Unassigned", "未分配", "할당되지 않음")
                                    .to_owned()
                            })
                        });
                    let combo = egui::ComboBox::from_id_salt(("expression_key", row.key.label()))
                        .selected_text(selected_label)
                        .width(ui.available_width().min(320.0))
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    row.expression.is_none(),
                                    lang.pick(
                                        "未割り当て",
                                        "Unassigned",
                                        "未分配",
                                        "할당되지 않음",
                                    ),
                                )
                                .clicked()
                                && row.expression.is_some()
                                && let Some(target) = target.clone()
                            {
                                state.emit(UiAction::AssignExpressionKey {
                                    target,
                                    key: row.key,
                                    expression: None,
                                });
                            }
                            // A saved binding that no longer exists in the
                            // current catalog stays visible, never silently
                            // replaced with a different expression.
                            if let Some(current) = &row.expression
                                && !vm
                                    .expression
                                    .entries
                                    .iter()
                                    .any(|entry| &entry.id == current)
                            {
                                ui.add_enabled(
                                    false,
                                    egui::Button::selectable(
                                        true,
                                        format!(
                                            "{current} — {}",
                                            lang.pick(
                                                "利用できません",
                                                "Unavailable",
                                                "不可用",
                                                "사용할 수 없음"
                                            )
                                        ),
                                    ),
                                );
                            }
                            for entry in &vm.expression.entries {
                                let label = format!(
                                    "{}{}",
                                    expression_entry_name(entry, lang),
                                    expression_availability_suffix(&entry.availability, lang)
                                );
                                let selected = row.expression.as_deref() == Some(entry.id.as_str());
                                if ui
                                    .add_enabled(
                                        entry.availability.is_ready(),
                                        egui::Button::selectable(selected, label),
                                    )
                                    .clicked()
                                    && let Some(target) = target.clone()
                                {
                                    state.emit(UiAction::AssignExpressionKey {
                                        target,
                                        key: row.key,
                                        expression: Some(entry.id.clone()),
                                    });
                                }
                            }
                        });
                    let _ = combo;
                    if row.selected {
                        ui.label(
                            RichText::new(lang.pick("選択中", "Selected", "已选中", "선택됨"))
                                .small()
                                .weak(),
                        );
                    }
                });
            }
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button(lang.pick("表情を解除", "Clear Expression", "取消表情", "표정 해제"))
                    .clicked()
                    && let Some(generation) = vm
                        .expression
                        .target
                        .as_ref()
                        .map(|target| target.generation)
                {
                    state.emit(UiAction::ClearManualExpression { generation });
                }
                if ui
                    .button(lang.pick(
                        "初期割り当てに戻す",
                        "Restore Default Bindings",
                        "恢复默认绑定",
                        "기본 키 할당 복원",
                    ))
                    .clicked()
                    && let Some(target) = target.clone()
                {
                    state.emit(UiAction::ResetExpressionBindings { target });
                }
            });
            ui.label(
                RichText::new(lang.pick(
                    "キーを押すと表情を選択し、同じキーをもう一度押すと解除します。文字入力中やダイアログ表示中は反応しません。",
                    "Press a key to select an expression. Press the same key again to clear it. Expression keys are inactive while typing or using dialogs.",
                    "按键可选择表情，再次按下同一按键可取消。输入文字或使用对话框时，表情按键不会生效。",
                    "키를 눌러 표정을 선택하고 같은 키를 다시 누르면 해제합니다. 텍스트 입력 중이거나 대화상자를 사용하는 동안에는 표정 키가 작동하지 않습니다.",
                ))
                .small()
                .weak(),
            );
        },
    );
}

/// Studio list of assigned expressions. Clicking runs the same toggle action
/// as the physical key.
fn expression_status_section(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    if !vm.expression.has_catalog {
        return;
    }
    let Some(generation) = vm
        .expression
        .target
        .as_ref()
        .map(|target| target.generation)
    else {
        return;
    };
    section(
        ui,
        lang.pick("表情キー", "Expression keys", "表情按键", "표정 키"),
        |ui| {
            let mut any = false;
            for row in &vm.expression.bindings {
                let Some(id) = &row.expression else {
                    continue;
                };
                any = true;
                let name = vm
                    .expression
                    .entries
                    .iter()
                    .find(|entry| &entry.id == id)
                    .map_or_else(|| id.clone(), |entry| expression_entry_name(entry, lang));
                let text = if row.selected {
                    format!(
                        "● {}  {} ({})",
                        row.key.label(),
                        name,
                        lang.pick("選択中", "Selected", "已选中", "선택됨")
                    )
                } else {
                    format!("{}  {name}", row.key.label())
                };
                if ui
                    .add(egui::Button::selectable(row.selected, text))
                    .clicked()
                {
                    state.emit(UiAction::ToggleExpressionKey {
                        generation,
                        key: row.key,
                    });
                }
            }
            if !any {
                ui.label(
                    RichText::new(lang.pick(
                        "割り当て済みの表情はありません。",
                        "No expressions are assigned.",
                        "尚未分配表情。",
                        "할당된 표정이 없습니다.",
                    ))
                    .weak(),
                );
            }
        },
    );
}

/// Translates an egui physical key into the fixed expression key, if bound.
#[must_use]
fn expression_key_from_egui(key: egui::Key) -> Option<ExpressionKey> {
    use egui::Key;
    Some(match key {
        Key::Num1 => ExpressionKey::Digit1,
        Key::Num2 => ExpressionKey::Digit2,
        Key::Num3 => ExpressionKey::Digit3,
        Key::Num4 => ExpressionKey::Digit4,
        Key::Num5 => ExpressionKey::Digit5,
        Key::Num6 => ExpressionKey::Digit6,
        Key::Num7 => ExpressionKey::Digit7,
        Key::Num8 => ExpressionKey::Digit8,
        Key::Num9 => ExpressionKey::Digit9,
        Key::Num0 => ExpressionKey::Digit0,
        Key::Q => ExpressionKey::KeyQ,
        Key::W => ExpressionKey::KeyW,
        Key::E => ExpressionKey::KeyE,
        Key::R => ExpressionKey::KeyR,
        Key::T => ExpressionKey::KeyT,
        Key::Y => ExpressionKey::KeyY,
        Key::U => ExpressionKey::KeyU,
        Key::I => ExpressionKey::KeyI,
        Key::O => ExpressionKey::KeyO,
        Key::P => ExpressionKey::KeyP,
        Key::A => ExpressionKey::KeyA,
        Key::S => ExpressionKey::KeyS,
        Key::D => ExpressionKey::KeyD,
        Key::F => ExpressionKey::KeyF,
        Key::G => ExpressionKey::KeyG,
        Key::H => ExpressionKey::KeyH,
        Key::J => ExpressionKey::KeyJ,
        Key::K => ExpressionKey::KeyK,
        Key::L => ExpressionKey::KeyL,
        Key::Z => ExpressionKey::KeyZ,
        Key::X => ExpressionKey::KeyX,
        Key::C => ExpressionKey::KeyC,
        Key::V => ExpressionKey::KeyV,
        Key::B => ExpressionKey::KeyB,
        Key::N => ExpressionKey::KeyN,
        Key::M => ExpressionKey::KeyM,
        _ => return None,
    })
}

/// Emits expression toggles for new, unmodified physical key-downs, and a
/// manual clear for unmodified Escape/Space.
///
/// The gate mirrors the product contract: window focused, avatar ready, no
/// text edit / IME / popup / modal / file dialog owning the keyboard, no
/// modifiers, and no key repeat. Multiple key-downs are emitted in event
/// order; the avatar runtime's request system applies them in the same order.
pub(crate) fn expression_key_input(
    ctx: &egui::Context,
    vm: &UiViewModel,
    state: &mut UiState,
    dialog_active: bool,
) {
    if !vm.avatar.is_ready
        || vm.avatar.lifecycle != AvatarLifecycleState::Ready
        || dialog_active
        || vm.avatar_import_review.review.is_some()
    {
        return;
    }
    if ctx.egui_wants_keyboard_input() || ctx.any_popup_open() {
        return;
    }
    let Some(generation) = vm
        .expression
        .target
        .as_ref()
        .map(|target| target.generation)
    else {
        return;
    };
    let actions: Vec<UiAction> = ctx.input(|input| {
        if !input.focused {
            return Vec::new();
        }
        input
            .events
            .iter()
            .filter_map(|event| match event {
                egui::Event::Key {
                    physical_key: Some(key),
                    pressed: true,
                    repeat: false,
                    modifiers,
                    ..
                } if modifiers.is_none() => match key {
                    egui::Key::Escape | egui::Key::Space => {
                        Some(UiAction::ClearManualExpression { generation })
                    }
                    key => expression_key_from_egui(*key)
                        .map(|key| UiAction::ToggleExpressionKey { generation, key }),
                },
                _ => None,
            })
            .collect()
    });
    for action in actions {
        state.emit(action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_generation() -> vtuber_avatar::AvatarGeneration {
        vtuber_avatar::AvatarGeneration(1)
    }

    fn ready_expression_view_model() -> UiViewModel {
        let mut vm = UiViewModel::default();
        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = AvatarLifecycleState::Ready;
        vm.expression.target = Some(crate::actions::ModelActionTarget {
            model_id: "model".into(),
            generation: test_generation(),
        });
        vm
    }

    fn expression_key_event(
        key: egui::Key,
        repeat: bool,
        modifiers: egui::Modifiers,
    ) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: Some(key),
            pressed: true,
            repeat,
            modifiers,
        }
    }

    fn run_expression_key_input(
        vm: &UiViewModel,
        state: &mut UiState,
        input: egui::RawInput,
        dialog_active: bool,
    ) {
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(input, |ctx| {
            expression_key_input(ctx, vm, state, dialog_active)
        });
    }

    fn focused_input(events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            focused: true,
            events,
            ..Default::default()
        }
    }

    #[test]
    fn expression_key_input_emits_toggle_even_in_avatar_only_mode() {
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        state.set_controls_open(false);
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert_eq!(
            state.take_actions(),
            vec![UiAction::ToggleExpressionKey {
                generation: test_generation(),
                key: ExpressionKey::Digit1
            }]
        );
    }

    #[test]
    fn expression_key_input_escape_and_space_clear_the_manual_expression() {
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![
                expression_key_event(egui::Key::Escape, false, egui::Modifiers::NONE),
                expression_key_event(egui::Key::Space, false, egui::Modifiers::NONE),
            ]),
            false,
        );
        assert_eq!(
            state.take_actions(),
            vec![
                UiAction::ClearManualExpression {
                    generation: test_generation()
                },
                UiAction::ClearManualExpression {
                    generation: test_generation()
                },
            ]
        );
    }

    #[test]
    fn expression_key_input_ignores_modified_and_repeated_clear_keys() {
        let ctx = egui::Context::default();
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        // Frame 1: Escape goes down once and clears.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Escape,
                false,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert_eq!(state.take_actions().len(), 1);

        // Frame 2: the OS repeats the held Escape. egui marks the event as a
        // repeat, which must not clear again.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Escape,
                true,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(
            state.take_actions().is_empty(),
            "OS key repeat never clears again"
        );

        // A modified Space stays with the existing shortcut handling.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Space,
                false,
                egui::Modifiers::SHIFT,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(
            state.take_actions().is_empty(),
            "modifiers never clear the expression"
        );
    }

    #[test]
    fn expression_key_input_processes_multiple_keys_in_event_order() {
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![
                expression_key_event(egui::Key::Num1, false, egui::Modifiers::NONE),
                expression_key_event(egui::Key::Q, false, egui::Modifiers::NONE),
            ]),
            false,
        );
        assert_eq!(
            state.take_actions(),
            vec![
                UiAction::ToggleExpressionKey {
                    generation: test_generation(),
                    key: ExpressionKey::Digit1
                },
                UiAction::ToggleExpressionKey {
                    generation: test_generation(),
                    key: ExpressionKey::KeyQ
                },
            ]
        );
    }

    #[test]
    fn expression_key_input_ignores_os_key_repeat_and_key_up() {
        let ctx = egui::Context::default();
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        // Frame 1: the physical key goes down once.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert_eq!(state.take_actions().len(), 1);

        // Frame 2: the OS repeats the held key. egui marks the event as a
        // repeat, which must not toggle again.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(
            state.take_actions().is_empty(),
            "OS key repeat never toggles"
        );

        // Key-up alone never clears.
        let _ = ctx.run_ui(
            focused_input(vec![egui::Event::Key {
                key: egui::Key::Num1,
                physical_key: Some(egui::Key::Num1),
                pressed: false,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(state.take_actions().is_empty(), "key-up never clears");
    }

    #[test]
    fn expression_key_input_ignores_modifiers_and_unfocused_window() {
        let vm = ready_expression_view_model();

        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::SHIFT,
            )]),
            false,
        );
        assert!(
            state.take_actions().is_empty(),
            "Shift+1 must stay with the existing shortcut handling"
        );

        let mut state = UiState::default();
        let mut input = focused_input(vec![expression_key_event(
            egui::Key::Num1,
            false,
            egui::Modifiers::NONE,
        )]);
        input.focused = false;
        run_expression_key_input(&vm, &mut state, input, false);
        assert!(state.take_actions().is_empty(), "unfocused window is inert");
    }

    #[test]
    fn expression_key_input_ignores_dialog_and_license_modal() {
        let mut vm = ready_expression_view_model();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            true,
        );
        assert!(state.take_actions().is_empty(), "file dialog owns input");

        vm.avatar_import_review.review = Some(super::super::import_review::review_fixture());
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert!(state.take_actions().is_empty(), "license modal owns input");
    }

    #[test]
    fn expression_key_input_defers_to_a_focused_text_edit() {
        let ctx = egui::Context::default();
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        let mut text = String::new();
        // First frame registers and focuses the text edit.
        let _ = ctx.run_ui(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.text_edit_singleline(&mut text).request_focus();
            });
        });
        assert!(ctx.egui_wants_keyboard_input());
        let input = focused_input(vec![expression_key_event(
            egui::Key::Num1,
            false,
            egui::Modifiers::NONE,
        )]);
        let _ = ctx.run_ui(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.text_edit_singleline(&mut text);
            });
            expression_key_input(ctx, &vm, &mut state, false);
        });
        assert!(
            state.take_actions().is_empty(),
            "typing in a text field must not trigger expressions"
        );
    }

    #[test]
    fn expression_key_input_requires_a_ready_avatar() {
        let mut vm = UiViewModel::default();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert!(state.take_actions().is_empty());

        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = AvatarLifecycleState::Loading;
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert!(state.take_actions().is_empty());
    }

    #[test]
    fn expression_ui_strings_exist_in_all_four_languages() {
        let entry = ExpressionEntryViewModel {
            id: "happy".into(),
            source_name: "happy".into(),
            kind: ExpressionKind::EmotionalPreset,
            availability: ExpressionAvailability::Ready,
        };
        let custom = ExpressionEntryViewModel {
            id: "笑顔".into(),
            source_name: "笑顔".into(),
            kind: ExpressionKind::Tracking,
            availability: ExpressionAvailability::Unsupported {
                reason: "test".into(),
            },
        };
        for lang in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            assert!(!expression_entry_name(&entry, lang).is_empty());
            let custom_label = expression_entry_name(&custom, lang);
            assert!(custom_label.contains("笑顔"), "custom names stay verbatim");
            assert!(!expression_availability_suffix(&custom.availability, lang).is_empty());
            assert!(
                expression_availability_suffix(&ExpressionAvailability::Ready, lang).is_empty()
            );
        }
    }

    #[test]
    fn expression_settings_section_renders_with_and_without_a_catalog() {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let mut vm = ready_expression_view_model();
        // Without a catalog: the load-model message renders.
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            expression_keys_page(ui, &vm, &mut state, UiLanguage::Ja);
        });

        // With a catalog: a disabled unavailable candidate and a saved
        // unknown binding render without panicking.
        vm.expression.has_catalog = true;
        vm.expression.entries = vec![ExpressionEntryViewModel {
            id: "happy".into(),
            source_name: "happy".into(),
            kind: ExpressionKind::EmotionalPreset,
            availability: ExpressionAvailability::Empty,
        }];
        vm.expression.bindings = vec![crate::ui_model::ExpressionKeyBindingViewModel {
            key: ExpressionKey::Digit1,
            expression: Some("vanished".into()),
            selected: false,
        }];
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            expression_keys_page(ui, &vm, &mut state, UiLanguage::Ja);
        });
    }

    #[test]
    fn expression_key_mapping_covers_the_fixed_36_keys() {
        for key in ExpressionKey::ALL {
            let egui_key = match key {
                ExpressionKey::Digit1 => egui::Key::Num1,
                ExpressionKey::Digit2 => egui::Key::Num2,
                ExpressionKey::Digit3 => egui::Key::Num3,
                ExpressionKey::Digit4 => egui::Key::Num4,
                ExpressionKey::Digit5 => egui::Key::Num5,
                ExpressionKey::Digit6 => egui::Key::Num6,
                ExpressionKey::Digit7 => egui::Key::Num7,
                ExpressionKey::Digit8 => egui::Key::Num8,
                ExpressionKey::Digit9 => egui::Key::Num9,
                ExpressionKey::Digit0 => egui::Key::Num0,
                ExpressionKey::KeyQ => egui::Key::Q,
                ExpressionKey::KeyW => egui::Key::W,
                ExpressionKey::KeyE => egui::Key::E,
                ExpressionKey::KeyR => egui::Key::R,
                ExpressionKey::KeyT => egui::Key::T,
                ExpressionKey::KeyY => egui::Key::Y,
                ExpressionKey::KeyU => egui::Key::U,
                ExpressionKey::KeyI => egui::Key::I,
                ExpressionKey::KeyO => egui::Key::O,
                ExpressionKey::KeyP => egui::Key::P,
                ExpressionKey::KeyA => egui::Key::A,
                ExpressionKey::KeyS => egui::Key::S,
                ExpressionKey::KeyD => egui::Key::D,
                ExpressionKey::KeyF => egui::Key::F,
                ExpressionKey::KeyG => egui::Key::G,
                ExpressionKey::KeyH => egui::Key::H,
                ExpressionKey::KeyJ => egui::Key::J,
                ExpressionKey::KeyK => egui::Key::K,
                ExpressionKey::KeyL => egui::Key::L,
                ExpressionKey::KeyZ => egui::Key::Z,
                ExpressionKey::KeyX => egui::Key::X,
                ExpressionKey::KeyC => egui::Key::C,
                ExpressionKey::KeyV => egui::Key::V,
                ExpressionKey::KeyB => egui::Key::B,
                ExpressionKey::KeyN => egui::Key::N,
                ExpressionKey::KeyM => egui::Key::M,
            };
            assert_eq!(expression_key_from_egui(egui_key), Some(key));
        }
        assert_eq!(expression_key_from_egui(egui::Key::Escape), None);
        assert_eq!(expression_key_from_egui(egui::Key::Space), None);
        assert_eq!(expression_key_from_egui(egui::Key::F1), None);
    }
}
