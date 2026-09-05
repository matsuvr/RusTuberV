//! Temporary performance probe for `bind_expressions` steady-state cost
//! (untracked; removed after measurement).

use std::hint::black_box;
use std::time::Instant;

use bevy::prelude::*;
use bevy_vrm1::vrm::expressions::test_support::{
    default_override_settings, ExpressionCategoryTag, RetargetExpressionNodes,
    BindExpressionNode, ExpressionCategory,
};
use bevy_vrm1::VrmExpressionPlugin;

#[test]
fn perf_probe_bind_expressions() {
    let mut app = App::new();
    app.add_plugins(VrmExpressionPlugin);
    let mesh_entity = app.world_mut().spawn(MorphWeights::new(vec![0.0; 4], None).unwrap()).id();
    for _ in 0..52 {
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.3, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 0.01,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            default_override_settings(),
        ));
    }
    app.update();

    let iterations = 2_000;
    let start = Instant::now();
    for _ in 0..iterations {
        app.update();
        black_box(());
    }
    let per_frame = start.elapsed() / iterations;
    println!("bind_expressions_steady_frame: {per_frame:?}");
}
