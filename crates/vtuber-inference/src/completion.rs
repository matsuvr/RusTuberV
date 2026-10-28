//! Shared completion handling for face and Pose inference workers.

use vtuber_core::{WorkerHandle, WorkerResult};

use crate::{
    FailureStage, InferenceError, InferenceWorkerResult, InferenceWorkerState, SharedStatus,
};

/// Takes and joins only a worker which has already exited.
///
/// A running worker returns `None` immediately. Completed model/runtime
/// failures and panics remain in the shared status and are returned once.
///
/// # Errors
/// The returned result contains the retained inference error or `WorkerPanicked`.
pub fn reap_finished_worker(
    worker: &mut Option<WorkerHandle<InferenceWorkerResult>>,
    status: &SharedStatus,
) -> Option<Result<InferenceWorkerResult, InferenceError>> {
    if !worker.as_ref().is_some_and(WorkerHandle::is_finished) {
        return None;
    }
    let worker = worker.take()?;
    let result = finish_worker_join(worker.join(), status);
    let status = match status.lock() {
        Ok(status) => status,
        Err(_) => return Some(Err(InferenceError::StatusPoisoned)),
    };
    if status.state == InferenceWorkerState::Failed
        && let Some(failure) = &status.last_failure
    {
        return Some(Err(failure.error.clone()));
    }
    Some(result)
}

/// Retains a panic from either worker in the same typed failure status.
///
/// # Errors
/// Returns `WorkerPanicked` when the join reports a panic.
pub fn finish_worker_join(
    result: WorkerResult<InferenceWorkerResult>,
    status: &SharedStatus,
) -> Result<InferenceWorkerResult, InferenceError> {
    match result {
        WorkerResult::Completed(result) => {
            if let Some(error) = &result.failure {
                return Err(error.clone());
            }
            let status = status.lock().map_err(|_| InferenceError::StatusPoisoned)?;
            if let Some(failure) = &status.last_failure
                && matches!(
                    failure.error,
                    InferenceError::SlotPoisoned(_) | InferenceError::StatusPoisoned
                )
            {
                return Err(failure.error.clone());
            }
            Ok(result)
        }
        WorkerResult::Panicked => {
            if let Ok(mut status) = status.lock() {
                status.record_failure(
                    vtuber_core::monotonic_now(),
                    FailureStage::WorkerPanic,
                    InferenceError::WorkerPanicked,
                );
            }
            Err(InferenceError::WorkerPanicked)
        }
    }
}
