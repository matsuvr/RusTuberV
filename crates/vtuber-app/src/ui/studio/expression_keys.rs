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
pub(super) fn expression_entry_name(entry: &ExpressionEntryViewModel, lang: UiLanguage) -> String {
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
pub(super) fn expression_availability_suffix(
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
pub(crate) fn expression_key_from_egui(key: egui::Key) -> Option<ExpressionKey> {
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
