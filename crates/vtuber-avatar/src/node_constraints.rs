//! Authored helper constraints evaluated after the admitted arm composition.
//!
//! Upstream initialization queues one component insertion per destination and
//! loses siblings sharing a source. Bind the complete authored list once.

use bevy::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

/// Rotation transfer declared by `VRMC_node_constraint`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeConstraintKind {
    /// Copy the source's rest-relative local rotation.
    Rotation,
    /// Copy only twist around the destination's local axis.
    Roll(Vec3),
    /// Aim the destination's local axis at the source's world position.
    Aim(Vec3),
}

/// One helper constraint from the managed VRM document.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceNodeConstraint {
    /// Source scene node name.
    pub source: String,
    /// Destination scene node name.
    pub destination: String,
    /// Authored constraint and axis.
    pub kind: NodeConstraintKind,
    /// Authored interpolation weight.
    pub weight: f32,
}

/// Complete authored helper constraints carried on the avatar root.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct SourceNodeConstraints(pub Vec<SourceNodeConstraint>);

/// A resolved helper destination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeConstraintDestination {
    /// Destination scene entity.
    pub destination: Entity,
    /// Authored constraint and axis.
    pub kind: NodeConstraintKind,
    /// Authored interpolation weight.
    pub weight: f32,
}

/// Every resolved destination grouped by its source, without overwriting siblings.
#[derive(Component, Clone, Debug, Default)]
pub struct NodeConstraintBindings(pub HashMap<Entity, Vec<NodeConstraintDestination>>);

/// Reads helper constraints from the common VRM 1.0 managed document.
///
/// # Errors
/// Returns the destination's node index when its source, axis or weight is invalid.
pub fn parse_source_node_constraints(document: &Value) -> Result<SourceNodeConstraints, usize> {
    let mut constraints = Vec::new();
    let Some(nodes) = document.get("nodes").and_then(Value::as_array) else {
        return Ok(SourceNodeConstraints(constraints));
    };
    let name = |index: usize| {
        nodes.get(index).map(|node| {
            node.get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("GltfNode{index}"))
        })
    };
    for (index, node) in nodes.iter().enumerate() {
        let Some(constraint) = node.pointer("/extensions/VRMC_node_constraint/constraint") else {
            continue;
        };
        let (kind, entry) = if let Some(entry) = constraint.get("rotation") {
            (NodeConstraintKind::Rotation, entry)
        } else if let Some(entry) = constraint.get("roll") {
            let axis = match entry.get("rollAxis").and_then(Value::as_str) {
                Some("X") => Vec3::X,
                Some("Y") => Vec3::Y,
                Some("Z") => Vec3::Z,
                _ => return Err(index),
            };
            (NodeConstraintKind::Roll(axis), entry)
        } else if let Some(entry) = constraint.get("aim") {
            let axis = match entry.get("aimAxis").and_then(Value::as_str) {
                Some("PositiveX") => Vec3::X,
                Some("NegativeX") => Vec3::NEG_X,
                Some("PositiveY") => Vec3::Y,
                Some("NegativeY") => Vec3::NEG_Y,
                Some("PositiveZ") => Vec3::Z,
                Some("NegativeZ") => Vec3::NEG_Z,
                _ => return Err(index),
            };
            (NodeConstraintKind::Aim(axis), entry)
        } else {
            return Err(index);
        };
        let source = entry
            .get("source")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(name)
            .ok_or(index)?;
        let weight = entry
            .get("weight")
            .map(Value::as_f64)
            .unwrap_or(Some(1.0))
            .ok_or(index)? as f32;
        if !(0.0..=1.0).contains(&weight) {
            return Err(index);
        }
        constraints.push(SourceNodeConstraint {
            source,
            destination: name(index).ok_or(index)?,
            kind,
            weight,
        });
    }
    Ok(SourceNodeConstraints(constraints))
}

impl SourceNodeConstraints {
    /// Resolves the complete list against this avatar's scene names.
    #[must_use]
    pub fn bind(&self, nodes: &HashMap<String, Entity>) -> NodeConstraintBindings {
        let mut bindings = HashMap::<Entity, Vec<NodeConstraintDestination>>::new();
        for constraint in &self.0 {
            if let Some((&source, &destination)) = nodes
                .get(&constraint.source)
                .zip(nodes.get(&constraint.destination))
            {
                bindings
                    .entry(source)
                    .or_default()
                    .push(NodeConstraintDestination {
                        destination,
                        kind: constraint.kind,
                        weight: constraint.weight,
                    });
            }
        }
        NodeConstraintBindings(bindings)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn shared_source_preserves_every_sleeve_destination_and_default_weight() {
        let document = serde_json::json!({"nodes": [
            {"name": "Elbow"},
            {"name": "Sleeve", "extensions": {"VRMC_node_constraint": {"constraint": {
                "aim": {"source": 0, "aimAxis": "PositiveX"}
            }}}},
            {"extensions": {"VRMC_node_constraint": {"constraint": {
                "aim": {"source": 0, "aimAxis": "NegativeX", "weight": 0.5}
            }}}}
        ]});
        let source = parse_source_node_constraints(&document).unwrap();
        let mut world = World::new();
        let elbow = world.spawn_empty().id();
        let sleeve = world.spawn_empty().id();
        let unnamed_destination = world.spawn_empty().id();
        let bound = source.bind(&HashMap::from([
            ("Elbow".into(), elbow),
            ("Sleeve".into(), sleeve),
            ("GltfNode2".into(), unnamed_destination),
        ]));
        assert_eq!(bound.0[&elbow].len(), 2);
        assert_eq!(bound.0[&elbow][0].destination, sleeve);
        assert_eq!(bound.0[&elbow][0].weight, 1.0);
        assert_eq!(bound.0[&elbow][1].destination, unnamed_destination);
        assert_eq!(
            bound.0[&elbow][1].kind,
            NodeConstraintKind::Aim(Vec3::NEG_X)
        );
        assert_eq!(bound.0[&elbow][1].weight, 0.5);
    }

    #[test]
    fn invalid_source_axis_or_weight_returns_the_destination_index() {
        for entry in [
            serde_json::json!({"source": 2, "rollAxis": "X"}),
            serde_json::json!({"source": 0, "rollAxis": "W"}),
            serde_json::json!({"source": 0, "rollAxis": "X", "weight": 1.1}),
        ] {
            let document = serde_json::json!({"nodes": [
                {}, {"extensions": {"VRMC_node_constraint": {"constraint": {"roll": entry}}}}
            ]});
            assert_eq!(parse_source_node_constraints(&document), Err(1));
        }
    }
}
