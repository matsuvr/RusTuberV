//! Typed task completion: display text never determines process control.

use std::ffi::OsString;
use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum TaskOutcome {
    Completed,
    Help,
    NotRun { reason: String, exit_code: i32 },
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct TaskError {
    message: String,
    exit_code: i32,
}

impl TaskError {
    pub(super) fn new(message: impl Into<String>, exit_code: i32) -> Self {
        Self {
            message: message.into(),
            exit_code,
        }
    }
}

impl From<String> for TaskError {
    fn from(message: String) -> Self {
        Self::new(message, 1)
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TaskError {}

pub(super) type TaskResult = Result<TaskOutcome, TaskError>;

pub(super) fn task_exit_code(result: &TaskResult) -> i32 {
    match result {
        Ok(TaskOutcome::Completed | TaskOutcome::Help) => 0,
        Ok(TaskOutcome::NotRun { exit_code, .. }) => *exit_code,
        Err(error) => error.exit_code,
    }
}

pub(super) fn completed(result: Result<(), String>) -> TaskResult {
    result
        .map(|()| TaskOutcome::Completed)
        .map_err(TaskError::from)
}

pub(super) fn decode_cli_args(
    args: impl IntoIterator<Item = OsString>,
) -> Result<Vec<String>, TaskError> {
    args.into_iter()
        .map(|arg| {
            arg.into_string().map_err(|arg| {
                TaskError::new(format!("CLI argument is not valid Unicode: {arg:?}"), 1)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_do_not_depend_on_messages() {
        assert_eq!(task_exit_code(&Ok(TaskOutcome::Completed)), 0);
        assert_eq!(task_exit_code(&Ok(TaskOutcome::Help)), 0);
        for text in [
            "unavailable",
            "別の説明",
            "NOT RUN: misleading text",
            "help requested",
        ] {
            assert_eq!(
                task_exit_code(&Ok(TaskOutcome::NotRun {
                    reason: text.into(),
                    exit_code: 2
                })),
                2
            );
            assert_eq!(task_exit_code(&Err(TaskError::new(text, 7))), 7);
        }
    }

    #[test]
    fn unicode_arguments_round_trip() {
        assert_eq!(
            decode_cli_args([OsString::from("モデル.vrm")]).unwrap(),
            ["モデル.vrm"]
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_arguments_are_errors_not_panics_or_lossy_paths() {
        use std::os::unix::ffi::OsStringExt;
        assert!(decode_cli_args([OsString::from_vec(vec![0xff])]).is_err());
    }
}
