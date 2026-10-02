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
    let status = status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        WorkerResult::Completed(result) => Ok(result),
        WorkerResult::Panicked => {
            status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .record_failure(FailureStage::WorkerPanic, InferenceError::WorkerPanicked);
            Err(InferenceError::WorkerPanicked)
        }
    }
}
