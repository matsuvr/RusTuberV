//! RRT-Connect (Kuffner & LaValle, ICRA 2000) for contact-trapped IK branches.
//! Tree edges use the continuous ROM/skin certificate, never point samples.

use crate::{
    upper_limb::{ArmJoints, BOUNDS},
    upper_limb_body::{BodyCurve, BodyPose},
    upper_limb_path::{certify, interpolate},
    upper_limb_solver::{Problem, SolveStatus},
};
use rand::{RngExt, SeedableRng};

type Point = [f32; 19];
type Waypoint = ([Option<ArmJoints>; 2], Option<BodyPose>);

#[derive(Clone, Copy)]
struct Node {
    point: Point,
    parent: usize,
}

struct Space<'a, 'b> {
    problem: &'a Problem<'b>,
    from: [Option<ArmJoints>; 2],
    to: [Option<ArmJoints>; 2],
}

fn coordinates(pose: [Option<ArmJoints>; 2], progress: f32) -> Point {
    let mut point = [progress; 19];
    for ((q, (lo, hi)), value) in pose
        .into_iter()
        .flat_map(|p| p.map_or([0.0; 9], |p| p.angles))
        .zip(BOUNDS.into_iter().cycle())
        .zip(point.iter_mut())
    {
        *value = (q - lo) / (hi - lo);
    }
    point
}

fn distance(a: Point, b: Point) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

impl Space<'_, '_> {
    fn pose(&self, point: Point) -> Waypoint {
        if point == coordinates(self.from, 0.0) {
            return (self.from, self.problem.body_curve.map(|c| c.from.clone()));
        }
        if point == coordinates(self.to, 1.0) {
            return (self.to, self.problem.body_curve.map(|c| c.to.clone()));
        }
        let progress = point.last().copied().unwrap_or(0.0);
        let mut pose = interpolate(self.from, self.to, progress);
        for (arm, values) in pose.iter_mut().zip(point.chunks_exact(9)) {
            if let Some(arm) = arm {
                for ((angle, value), (lo, hi)) in arm.angles.iter_mut().zip(values).zip(BOUNDS) {
                    *angle = lo + value * (hi - lo);
                }
            }
        }
        (pose, self.problem.body_curve.map(|c| c.at(progress)))
    }

    fn clear(&self, a: Point, b: Point) -> bool {
        let (from, body_a) = self.pose(a);
        let (to, body_b) = self.pose(b);
        let curve = body_a.zip(body_b).map(|(from, to)| BodyCurve { from, to });
        let Ok(body) = curve.as_ref().map(|c| c.to.motions()).transpose() else {
            return false;
        };
        let problem = Problem {
            body: body.as_ref().unwrap_or(self.problem.body),
            body_curve: curve.as_ref(),
            ..*self.problem
        };
        // Most sampled configurations fail a joint bound; reject them before
        // posing skin. Then stop at the first collision instead of computing
        // the full distance/Jacobian input that only the SQP needs.
        let Ok(evaluation) = problem.kinematics(to) else {
            return false;
        };
        if !problem.feasible(&evaluation) {
            return false;
        }
        problem.geometry.pose_is_clear(&|bone| {
            evaluation
                .arms
                .iter()
                .flatten()
                .find_map(|a| a.motion.get(&bone).copied())
                .or_else(|| problem.body.get(&bone).copied())
        }) == Ok(true)
            && certify(&problem, from, to)
    }

    fn extend(&self, tree: &mut Vec<Node>, target: Point) -> Option<(usize, bool)> {
        let (parent, nearest) = tree.iter().enumerate().min_by(|(_, a), (_, b)| {
            distance(a.point, target).total_cmp(&distance(b.point, target))
        })?;
        let d = distance(nearest.point, target);
        if d == 0.0 {
            return Some((parent, true));
        }
        // A search resolution in normalized configuration space, not a
        // biomechanical limit. All accepted edges are continuously checked.
        let amount = (0.15 / d).min(1.0);
        let point = if amount == 1.0 {
            target
        } else {
            std::array::from_fn(|i| {
                nearest
                    .point
                    .get(i)
                    .zip(target.get(i))
                    .map_or(0.0, |(a, b)| a + (b - a) * amount)
            })
        };
        if !self.clear(nearest.point, point) {
            return None;
        }
        tree.push(Node { point, parent });
        Some((tree.len() - 1, amount == 1.0))
    }
}

fn branch(tree: &[Node], mut index: usize) -> Vec<Point> {
    let mut result = Vec::new();
    while let Some(node) = tree.get(index) {
        result.push(node.point);
        if node.parent == index {
            break;
        }
        index = node.parent;
    }
    result
}

pub(crate) fn connect(
    problem: &Problem<'_>,
    from: [Option<ArmJoints>; 2],
    to: [Option<ArmJoints>; 2],
) -> Result<Vec<Waypoint>, SolveStatus> {
    let space = Space { problem, from, to };
    let start = coordinates(from, 0.0);
    let goal = coordinates(to, 1.0);
    if space.clear(start, goal) {
        return Ok(vec![space.pose(goal)]);
    }
    let mut a = vec![Node {
        point: start,
        parent: 0,
    }];
    let mut b = vec![Node {
        point: goal,
        parent: 0,
    }];
    // Reproducible sampling aids diagnosis. No OS entropy, anatomical
    // coefficients or side-specific branch preference are introduced.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(248);
    let mut reversed = false;
    for _ in 0..2048 {
        let sample = std::array::from_fn(|_| rng.random::<f32>());
        if let Some((end_a, _)) = space.extend(&mut a, sample)
            && let Some(target) = a.get(end_a).map(|n| n.point)
        {
            while let Some((end_b, reached)) = space.extend(&mut b, target) {
                if reached {
                    let mut first = branch(&a, end_a);
                    first.reverse();
                    first.extend(branch(&b, end_b).into_iter().skip(1));
                    if reversed {
                        first.reverse();
                    }
                    // Standard collision-checked shortcut smoothing removes
                    // random exploration detours before display.
                    let mut checked = vec![(start, goal)];
                    for _ in 0..32 {
                        // The only shortcut of a three-node route is the
                        // direct edge already rejected before tree growth.
                        if first.len() <= 3 {
                            break;
                        }
                        let i = rng.random_range(0..first.len() - 2);
                        let j = rng.random_range(i + 2..first.len());
                        if let Some((&a, &b)) = first.get(i).zip(first.get(j)) {
                            if checked.contains(&(a, b)) {
                                continue;
                            }
                            checked.push((a, b));
                            if space.clear(a, b) {
                                first.drain(i + 1..j);
                            }
                        }
                    }
                    return Ok(first.into_iter().skip(1).map(|p| space.pose(p)).collect());
                }
            }
        }
        std::mem::swap(&mut a, &mut b);
        reversed = !reversed;
    }
    Err(SolveStatus::BlockedPath)
}
