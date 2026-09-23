//! VRM 0.x/1.0 compatibility gate for the shared upstream runtime.
//!
//! This module introspects a loaded VRM model and records which runtime
//! capabilities are present. It is used by the compatibility runner and by
//! unit tests to guard against upstream behaviour changes at the pinned tag.
//!
//! Generation and import warnings are source records: the import pipeline
//! converts VRM 0.x sources into VRM 1.0-shaped managed copies, so the
//! runtime boundary itself no longer reports a generation. The expected
//! generation and the preflight warnings travel on the avatar root
//! (`ExpectedVrmGeneration`, [`VrmSourceWarnings`]).

use bevy::prelude::*;
use bevy_vrm1::prelude::*;

use crate::capabilities::PerfectSyncCapabilities;
use crate::load::ExpectedVrmGeneration;
use crate::vrm0::VrmCompatibilityWarning;

/// Plugin that installs compatibility-report systems.
#[derive(Default)]
pub struct VrmCompatibilityPlugin;

impl Plugin for VrmCompatibilityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VrmCompatibilityReport>()
            .add_systems(Update, inspect_initialized_vrm)
            .add_systems(Update, log_compatibility_report);
    }
}

/// Import warnings carried on the avatar root for the compatibility report.
///
/// The runtime only sees converted content, so the source diagnostics travel
/// here instead of living on a runtime component.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct VrmSourceWarnings(pub Vec<VrmCompatibilityWarning>);

/// Bone capability tuple used when querying a freshly-initialized VRM.
type InitializedVrmBones<'w, 's> = (
    Entity,
    Option<&'static ExpectedVrmGeneration>,
    Option<&'static VrmSourceWarnings>,
    Option<&'static HeadBoneEntity>,
    Option<&'static NeckBoneEntity>,
    Option<&'static LeftEyeBoneEntity>,
    Option<&'static RightEyeBoneEntity>,
    Option<&'static ExpressionEntityMap>,
    Option<&'static LookAt>,
    Option<&'static BodyTracking>,
);

/// Resource populated once a VRM has been inspected.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct VrmCompatibilityReport {
    /// Active runtime root owning this report.
    pub root: Option<Entity>,
    /// Generation recorded by app-side inspection at import.
    pub generation: Option<ExpectedVrmGeneration>,
    /// Whether a `Vrm` component was observed.
    pub vrm_loaded: bool,
    /// Whether the `Initialized` marker was observed.
    pub initialized: bool,
    /// Whether the head bone entity was found.
    pub has_head: bool,
    /// Whether the neck bone entity was found.
    pub has_neck: bool,
    /// Whether the left eye bone entity was found.
    pub has_left_eye: bool,
    /// Whether the right eye bone entity was found.
    pub has_right_eye: bool,
    /// Expression preset names discovered on the model.
    pub expressions: Vec<String>,
    /// ARKit52/Perfect Sync capability discovered on the model.
    pub perfect_sync: PerfectSyncCapabilities,
    /// Whether a `LookAt` component is present on the root.
    pub has_look_at_component: bool,
    /// Whether a `BodyTracking` component is present on the root.
    pub has_body_tracking_component: bool,
    /// Number of `SpringRoot` components found (proxy for SpringBone presence).
    pub spring_root_count: usize,
    /// Import warnings from the active source generation.
    pub warnings: Vec<VrmCompatibilityWarning>,
}

impl VrmCompatibilityReport {
    /// Returns `true` if the model is usable for the VTuber MVP.
    ///
    /// MVP requires at least a head bone and either per-eye blink or a
    /// combined blink expression. It does **not** require `LookAt` or an
    /// already-present `BodyTracking` component during raw runtime
    /// initialization; the application adds direct-pose body tracking after
    /// humanoid binding succeeds.
    #[must_use]
    pub fn is_mvp_capable(&self) -> bool {
        self.vrm_loaded && self.has_head && self.has_any_blink()
    }

    /// Returns `true` if either combined or per-eye blink is available.
    #[must_use]
    pub fn has_any_blink(&self) -> bool {
        self.has_expression("blink")
            || (self.has_expression("blinkLeft") && self.has_expression("blinkRight"))
    }

    /// Returns `true` if the named expression preset is available.
    #[must_use]
    pub fn has_expression(&self, name: &str) -> bool {
        self.expressions
            .iter()
            .any(|e| e.eq_ignore_ascii_case(name))
    }
}

fn inspect_initialized_vrm(
    mut report: ResMut<VrmCompatibilityReport>,
    vrms: Query<InitializedVrmBones, (With<Vrm>, Added<Initialized>)>,
    all_vrms: Query<Entity, With<Vrm>>,
    spring_roots: Query<&SpringRoot>,
) {
    if report.root.is_some_and(|root| !all_vrms.contains(root)) {
        *report = VrmCompatibilityReport::default();
    }
    for (
        entity,
        expected_generation,
        source_warnings,
        head,
        neck,
        left_eye,
        right_eye,
        expression_map,
        look_at,
        body_tracking,
    ) in vrms.iter()
    {
        report.root = Some(entity);
        report.vrm_loaded = true;
        report.initialized = true;
        report.generation = expected_generation.copied();
        report.has_head = head.is_some();
        report.has_neck = neck.is_some();
        report.has_left_eye = left_eye.is_some();
        report.has_right_eye = right_eye.is_some();
        report.has_look_at_component = look_at.is_some();
        report.has_body_tracking_component = body_tracking.is_some();
        report.spring_root_count = spring_roots.iter().count();
        // The runner inspects raw runtime initialization, which runs before
        // the application bind step records per-expression statuses.
        // Presence in the map implies effectiveness here: the import
        // preflight validates every node, mesh, and morph reference before a
        // model reaches the runtime.
        report.perfect_sync =
            PerfectSyncCapabilities::from_map_with_effective(expression_map, |_| true);
        report.warnings = source_warnings
            .map(|warnings| warnings.0.clone())
            .unwrap_or_default();

        if let Some(map) = expression_map {
            report.expressions = map
                .0
                .keys()
                .map(|expr| expr.to_string())
                .collect::<Vec<_>>();
            report.expressions.sort();
        }
    }
}

fn log_compatibility_report(report: Res<VrmCompatibilityReport>) {
    if report.is_changed() && report.initialized {
        info!("VRM compatibility report: {:?}", report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvp_capable_requires_head_and_blink() {
        let mut report = VrmCompatibilityReport::default();
        assert!(!report.is_mvp_capable());

        report.vrm_loaded = true;
        report.has_head = true;
        assert!(!report.is_mvp_capable());

        report.expressions.push("blink".into());
        assert!(report.is_mvp_capable());
    }

    #[test]
    fn per_eye_blink_satisfies_blink() {
        let report = VrmCompatibilityReport {
            vrm_loaded: true,
            has_head: true,
            expressions: vec!["blinkLeft".into(), "blinkRight".into()],
            ..Default::default()
        };
        assert!(report.is_mvp_capable());
    }

    #[test]
    fn warnings_are_cleared_when_the_active_runtime_root_is_replaced() {
        let mut app = App::new();
        app.init_resource::<VrmCompatibilityReport>()
            .add_systems(Update, inspect_initialized_vrm);
        let root = app
            .world_mut()
            .spawn((
                Vrm,
                Initialized,
                ExpectedVrmGeneration::Vrm0,
                VrmSourceWarnings(vec![VrmCompatibilityWarning::new(
                    crate::vrm0::VrmCompatibilityWarningCode::EmptyLegacyExpressionName,
                    "old-root",
                )]),
            ))
            .id();
        app.update();
        assert_eq!(
            app.world().resource::<VrmCompatibilityReport>().root,
            Some(root)
        );
        assert_eq!(
            app.world()
                .resource::<VrmCompatibilityReport>()
                .warnings
                .len(),
            1
        );

        app.world_mut().entity_mut(root).despawn();
        app.update();
        let report = app.world().resource::<VrmCompatibilityReport>();
        assert_eq!(report.root, None);
        assert!(report.warnings.is_empty());
    }
}
