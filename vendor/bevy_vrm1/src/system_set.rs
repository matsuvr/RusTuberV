use bevy::prelude::SystemSet;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Ord, PartialOrd, Clone, Copy)]
pub enum VrmSystemSets {
    /// Head-relative `LookAt` binding processing after humanoid pose.
    GazeControl,

    /// Expression binding processing.
    Expressions,

    /// Manual transform propagation after Expressions.
    /// This makes bone `LookAt` changes visible to constraints.
    PropagateAfterExpressions,

    /// Node constraints processing.
    Constraints,

    /// Manual transform propagation after Constraints.
    /// This makes constrained transforms visible to `SpringBone`.
    PropagateAfterConstraints,

    /// This is used for spring bones.
    SpringBone,

    /// This is used to determine whether to send a [`RequestRedraw`](bevy::window::RequestRedraw).
    DetermineRedraw,
}
