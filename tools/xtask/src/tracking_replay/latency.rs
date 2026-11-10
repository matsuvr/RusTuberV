//! Measurements on recorded tracking ticks. All times are in the recording's epoch.
use super::{ObservationTick, Row, distribution};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Duration,
};
use vtuber_core::{FrameSeq, MonoTimeNs};

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    match args {
        [cmd, tape, out] if cmd == "make-return" => make_return(tape, out),
        [cmd, dir] if cmd == "stages" => {
            let rows = read_rows(dir)?;
            println!("{}", stages(&rows));
            Ok(())
        }
        [cmd, clip, before, after] if cmd == "compare" => {
            let base = read_rows(before)?;
            let changed = read_rows(after)?;
            let same_ticks = base.len() == changed.len() && base.iter().zip(&changed).all(|(a, b)| {
                a.video_seconds == b.video_seconds && a.frame_ms == b.frame_ms
            });
            if !same_ticks {
                return Err("comparison requires the same recorded ticks".into());
            }
            let lag = bilateral_lag(&base, &changed);
            let summary = json!({
                "baseline": summarize(clip, &base)?,
                "candidate": summarize(clip, &changed)?,
                "lag": lag,
            });
            println!("{summary}");
            Ok(())
        }
        _ => Err("hand-latency stages <live-run> | compare <raise|cross|return> <baseline-run> <candidate-run> | make-return <observations.json> <new-tape.json>".into()),
    }
}

fn summarize(clip: &str, rows: &[Row]) -> Result<Value, String> {
    let mut statuses = BTreeMap::new();
    for row in rows {
        *statuses
            .entry(row.solver.split_whitespace().next().unwrap_or(""))
            .or_insert(0) += 1;
    }
    let event = match clip {
        "raise" => raise_event(rows),
        "cross" => crossings_event(rows),
        "return" => return_events(rows),
        _ => return Err("clip must be raise, cross or return".into()),
    };
    let mut lengths = [(f64::INFINITY, f64::NEG_INFINITY); 4];
    for row in rows {
        for ((lo, hi), (a, b)) in lengths.iter_mut().zip([(0, 1), (1, 2), (3, 4), (4, 5)]) {
            if let Some((a, b)) = row.joints.get(a).zip(row.joints.get(b)) {
                let len = position(*a).distance(position(*b));
                *lo = lo.min(len);
                *hi = hi.max(len);
            }
        }
    }
    let variation = lengths
        .into_iter()
        .filter(|(lo, hi)| lo <= hi)
        .map(|(lo, hi)| 1000. * (hi - lo))
        .max_by(f64::total_cmp);
    let quiet_rms = (clip == "raise").then(|| {
        std::array::from_fn::<_, 6, _>(|joint| {
            let points: Vec<_> = rows
                .iter()
                .filter(|r| (34.0..35.0).contains(&r.video_seconds))
                .filter_map(|r| r.joints.get(joint).copied().map(position))
                .collect();
            if points.is_empty() {
                return None;
            }
            let n = points.len() as f64;
            let mean = points.iter().copied().sum::<bevy::math::DVec3>() / n;
            Some(1000. * (points.iter().map(|p| p.distance_squared(mean)).sum::<f64>() / n).sqrt())
        })
    });
    Ok(
        json!({"event":event,"solver_frames":statuses,"bone_length_range_mm":variation,"quiet_rms_mm":quiet_rms}),
    )
}

fn position((xyz, _): ([f32; 3], [f32; 4])) -> bevy::math::DVec3 {
    bevy::math::DVec3::from_array(xyz.map(f64::from))
}

fn raise_event(rows: &[Row]) -> Value {
    let bounds = ([54.06, 54.13], [54.4, 55.0], [54.1, 55.0]);
    let wrist = series(rows, |r| Some(bone_xy(r, 0)?[1]));
    let head = series(rows, |r| {
        let [x, y, z, w] = r
            .bones
            .iter()
            .find(|b| b.name == "head")?
            .global_delta
            .map(f64::from);
        Some(
            (2.0 * (w * z + x * y))
                .atan2(1.0 - 2.0 * (y * y + z * z))
                .to_degrees(),
        )
    });
    let target_times = |section: &str, field: &str| {
        event_times(
            &series(rows, |r| side(r, section, 0)?.get(field)?.get(1)?.as_f64()),
            bounds,
        )
    };
    let mut pose = Vec::new();
    let mut hand = Vec::new();
    let mut previous = None;
    for row in rows {
        let Some(a) = &row.arm_observation else {
            continue;
        };
        let Some(seq) = a.get("source_seq").and_then(Value::as_u64) else {
            continue;
        };
        if previous.replace(seq) == Some(seq) {
            continue;
        }
        let Some(trace) = a.get("trace") else {
            continue;
        };
        let Some(image) = trace.get("pose_image").and_then(|v| v.get(0)) else {
            continue;
        };
        let Some(shoulder) = image
            .get("shoulder")
            .and_then(|v| v.get(1))
            .and_then(Value::as_f64)
        else {
            continue;
        };
        for (dst, point) in [
            (&mut pose, image.get("wrist")),
            (&mut hand, trace.get("hand_image").and_then(|v| v.get(0))),
        ] {
            let y = point.and_then(|p| p.get(1)).and_then(Value::as_f64);
            dst.push((seq as f64 / 30., y.map(|v| shoulder - v)));
        }
    }
    let head_quiet = [3.0, Duration::from_millis(3140).as_secs_f64()];
    json!({
        "wrist": event_times(&wrist,bounds),
        "head": event_times(&head,(head_quiet,[3.65,4.3],[3.1,4.3])),
        "accepted": target_times("status","accepted_wrist"),
        "filtered": target_times("sides","wrist"),
        "pose_image": event_times(&pose,bounds),
        "hand_image": event_times(&hand,bounds),
    })
}

fn crossings_event(rows: &[Row]) -> Value {
    let x = series(rows, |r| {
        Some(f64::from(r.joints.get(2)?.0[0] - r.joints.get(5)?.0[0]))
    });
    let crossings: Vec<_> = x
        .windows(2)
        .filter_map(|pair| {
            let [(t0, Some(a)), (t1, Some(b))] = pair else {
                return None;
            };
            (*t1 <= 7. && a * b < 0.).then(|| t0 - a * (t1 - t0) / (b - a))
        })
        .collect();
    json!({"crossings_first_7s":crossings})
}

fn return_events(rows: &[Row]) -> Value {
    let mut result = serde_json::Map::new();
    for (label, start, end) in [("small", 3., 4.5), ("large", 4.5, 6.), ("return", 9., 12.)] {
        let centre = |window| -> Option<[f64; 2]> {
            let points: Vec<_> = rows
                .iter()
                .filter(|r| inside(r.video_seconds, window))
                .filter_map(|r| bone_xy(r, 0))
                .collect();
            Some([
                median(points.iter().map(|p| p[0]))?,
                median(points.iter().map(|p| p[1]))?,
            ])
        };
        let Some(([x, y], [u, v])) = centre([start - 0.2, start]).zip(centre([end - 0.3, end]))
        else {
            continue;
        };
        let length = (u - x).hypot(v - y);
        if length <= 1e-6 {
            continue;
        }
        let samples = series(rows, |r| {
            let [px, py] = bone_xy(r, 0)?;
            Some(((px - x) * (u - x) + (py - y) * (v - y)) / length)
        });
        let window = [start - 1. / 30., end];
        let reach = crossings(&samples, 0., length, window).map(|t| t.map(|t| (t - start) * 1000.));
        let peak = samples
            .iter()
            .filter(|(t, _)| inside(*t, window))
            .filter_map(|s| s.1)
            .fold(length, f64::max);
        result.insert(
            label.to_owned(),
            json!({"ms":reach,"travel_mm":length*1000.,"overshoot_mm":(peak-length)*1000.}),
        );
    }
    Value::Object(result)
}

fn read_rows(dir: &str) -> Result<Vec<Row>, String> {
    let file = File::open(Path::new(dir).join("frames.jsonl")).map_err(|e| e.to_string())?;
    BufReader::new(file)
        .lines()
        .map(|s| serde_json::from_str(&s.map_err(|e| e.to_string())?).map_err(|e| e.to_string()))
        .collect()
}

fn stages(rows: &[Row]) -> Value {
    let mut seen = BTreeSet::new();
    let mut samples: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for row in rows
        .iter()
        .filter(|r| r.video_seconds >= super::WARMUP_SECONDS)
    {
        let Some(a) = &row.arm_observation else {
            continue;
        };
        let Some(seq) = a.get("source_seq").and_then(Value::as_u64) else {
            continue;
        };
        if !seen.insert(seq) {
            continue;
        }
        let Some(trace) = a.get("trace").filter(|v| v.is_object()) else {
            continue;
        };
        let native = |key, i: usize| trace.get(key)?.get(i);
        let capture = a.get("captured_at");
        let started = trace.get("started_at");
        let finished = a.get("inference_finished_at");
        let received = a.get("sampled_at");
        for (stage, start, end) in [
            ("capture_to_worker_ms", capture, started),
            ("image_conversion_ms", started, native("pose_interval", 0)),
            (
                "pose_ms",
                native("pose_interval", 0),
                native("pose_interval", 1),
            ),
            (
                "pose_to_hand_ms",
                native("pose_interval", 1),
                native("hand_interval", 0),
            ),
            (
                "hand_ms",
                native("hand_interval", 0),
                native("hand_interval", 1),
            ),
            ("worker_total_ms", started, finished),
            ("finished_to_receive_ms", finished, received),
            ("capture_to_receive_ms", capture, received),
        ] {
            if let Some(ms) = elapsed_ms(start, end) {
                samples.entry(stage).or_default().push(ms);
            }
        }
    }
    Value::Object(
        samples
            .into_iter()
            .map(|(name, v)| (name.to_owned(), distribution(v.into_iter())))
            .collect(),
    )
}

fn elapsed_ms(start: Option<&Value>, end: Option<&Value>) -> Option<f64> {
    Some(end?.as_u64()?.checked_sub(start?.as_u64()?)? as f64 / 1e6)
}

fn side<'a>(row: &'a Row, section: &str, index: usize) -> Option<&'a Value> {
    row.arm_observation.as_ref()?.get(section)?.get(index)
}

// Sapphy, front view, default mirror. Swap chains and undo the X reflection.
fn bone_xy(row: &Row, anatomical_side: usize) -> Option<[f64; 2]> {
    let index = if anatomical_side == 0 { 3 } else { 0 };
    let shoulder = row.joints.get(index)?.0;
    let wrist = row.joints.get(index + 2)?.0;
    Some([
        f64::from(shoulder[0] - wrist[0]),
        f64::from(wrist[1] - shoulder[1]),
    ])
}

type Series = Vec<(f64, Option<f64>)>;
fn series(rows: &[Row], value: impl Fn(&Row) -> Option<f64>) -> Series {
    rows.iter().map(|r| (r.video_seconds, value(r))).collect()
}
fn inside(t: f64, [lo, hi]: [f64; 2]) -> bool {
    (lo..hi).contains(&t)
}
fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<_> = values.collect();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some((v.get(n.checked_sub(1)? / 2)? + v.get(n / 2)?) * 0.5)
}

fn event_times(
    samples: &[(f64, Option<f64>)],
    (quiet, peak, span): ([f64; 2], [f64; 2], [f64; 2]),
) -> [Option<f64>; 3] {
    let start = median(
        samples
            .iter()
            .filter(|(t, _)| inside(*t, quiet))
            .filter_map(|s| s.1),
    );
    let end = samples
        .iter()
        .filter(|(t, _)| inside(*t, peak))
        .filter_map(|s| s.1)
        .max_by(f64::total_cmp);
    start
        .zip(end)
        .map_or([None; 3], |(a, b)| crossings(samples, a, b, span))
}
fn crossings(
    samples: &[(f64, Option<f64>)],
    start: f64,
    end: f64,
    span: [f64; 2],
) -> [Option<f64>; 3] {
    [0.1, 0.5, 0.9].map(|level| {
        let threshold = start + (end - start) * level;
        samples.windows(2).find_map(|w| {
            let [(t, Some(a)), (u, Some(b))] = w else {
                return None;
            };
            (end > start
                && inside(*t, span)
                && inside(*u, span)
                && *a < threshold
                && *b >= threshold)
                .then(|| t + (threshold - a) * (u - t) / (b - a))
        })
    })
}
/// Same contiguous samples for both versions and all lags. Positive lag means
/// the displayed wrist follows the accepted observation. Centre each segment
/// separately so rest offsets cannot score as motion.
fn bilateral_lag(base: &[Row], changed: &[Row]) -> Value {
    use bevy::math::DVec2;
    type Sample = [DVec2; 3]; // accepted, baseline final, candidate final
    let mut segments: Vec<Vec<Sample>> = Vec::new();
    let Some(last) = base.last() else {
        return Value::Null;
    };
    for side_index in 0..2 {
        let at = |i| -> Option<Sample> {
            let a = base.get(i)?;
            let b = changed.get(i)?;
            for row in [a, b] {
                let weight = side(row, "sides", side_index)?
                    .get("weight")?
                    .get(0)?
                    .as_f64()?;
                if weight < 0.8
                    || !side(row, "status", side_index)?
                        .get("accepted_wrist")?
                        .is_array()
                {
                    return None;
                }
            }
            let paired = a
                .arm_observation
                .as_ref()?
                .get("trace")?
                .get("hand_image")?
                .get(side_index)?
                .is_array();
            if !paired {
                return None;
            }
            let p = side(a, "status", side_index)?.get("accepted_wrist")?;
            Some([
                DVec2::new(p.get(0)?.as_f64()?, p.get(1)?.as_f64()?),
                DVec2::from_array(bone_xy(a, side_index)?),
                DVec2::from_array(bone_xy(b, side_index)?),
            ])
        };
        let mut segment = Vec::new();
        let mut grid = 0usize;
        loop {
            let t = super::WARMUP_SECONDS + grid as f64 / 60.;
            if t >= last.video_seconds {
                break;
            }
            let j = base.partition_point(|r| r.video_seconds < t);
            let sample = j.checked_sub(1).and_then(|i| {
                let t0 = base.get(i)?.video_seconds;
                let t1 = base.get(j)?.video_seconds;
                let f = (t - t0) / (t1 - t0);
                let [ra, aa, ba] = at(i)?;
                let [rb, ab, bb] = at(j)?;
                Some([ra.lerp(rb, f), aa.lerp(ab, f), ba.lerp(bb, f)])
            });
            if let Some(s) = sample {
                segment.push(s)
            } else if !segment.is_empty() {
                segments.push(std::mem::take(&mut segment));
            }
            grid += 1;
        }
        if !segment.is_empty() {
            segments.push(segment);
        }
    }
    const MAX_LAG: usize = 36;
    segments.retain(|s| {
        let Some(input) = s.len().checked_sub(MAX_LAG).and_then(|n| s.get(..n)) else {
            return false;
        };
        let (lo, hi) = input.iter().map(|p| p[0]).fold(
            (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
            |(lo, hi), v| (lo.min(v), hi.max(v)),
        );
        input.len() >= 30 && (hi - lo).max_element() > 0.02
    });
    let estimate = |output: fn(&Sample) -> DVec2| {
        (0..=MAX_LAG).filter_map(|lag|{
            let mut dot=0.;let mut xx=0.; let mut yy=0.;
            for s in &segments {
                let n=s.len().checked_sub(MAX_LAG)?;
                let a=s.get(..n)?;let b=s.get(lag..lag+n)?;
                let mx=a.iter().map(|p|p[0]).sum::<DVec2>()/n as f64;
                let my=b.iter().map(output).sum::<DVec2>()/n as f64;
                for (p,q) in a.iter().zip(b) {
                    let x=p[0]-mx; let y=output(q)-my;
                    dot+=x.dot(y); xx+=x.length_squared(); yy+=y.length_squared();
                }
            }
            let norm=(xx*yy).sqrt();
            (norm>0.).then(||(lag,dot/norm))
        }).max_by(|a,b|a.1.total_cmp(&b.1)).map(|(lag,r)|json!({"ms":lag as f64*1000./60., "correlation":r,"segments":segments.len()}))
    };
    json!({"baseline":estimate(|p|p[1]),"candidate":estimate(|p|p[2])})
}

fn make_return(input: &str, out: &str) -> Result<(), String> {
    let ticks: Vec<ObservationTick> = serde_json::from_reader(BufReader::new(
        File::open(input).map_err(|e| e.to_string())?,
    ))
    .map_err(|e| e.to_string())?;
    let seed = ticks
        .iter()
        .filter_map(|tick| {
            let raw = tick.raw?;
            let pose = raw.observation?;
            (pose.left.hand.is_some() && pose.right.hand.is_some() && tick.face.is_some())
                .then_some((tick, raw.source_seq.0))
        })
        .min_by_key(|(_, seq)| seq.abs_diff(239))
        .ok_or("no seed frame with both hands and a face")?;
    let mut result = Vec::new();
    for j in 0..840u64 {
        let mut tick = seed.0.clone();
        let seq = j / 2;
        let now = MonoTimeNs(3_000_000_000 + j * 1_000_000_000 / 60);
        let captured = MonoTimeNs(2_985_000_000 + seq * 1_000_000_000 / 30);
        tick.seconds = j as f64 / 60.;
        tick.delta = Duration::from_nanos(16_666_667);
        tick.now = now;
        tick.pose_now = Some(now);
        tick.tracked = true;
        let raw = tick.raw.as_mut().ok_or("missing seed pose")?;
        raw.source_seq = FrameSeq(seq);
        raw.captured_at = captured;
        raw.inference_finished_at = MonoTimeNs(captured.0 + 13_000_000);
        raw.trace = None;
        let face = &mut tick.face.as_mut().ok_or("missing seed face")?.0;
        face.source_seq = raw.source_seq;
        face.captured_at = captured;
        face.produced_at = now;
        let arm = &mut raw.observation.as_mut().ok_or("missing seed arm")?.left;
        if (180..270).contains(&seq) {
            arm.hand = None;
        }
        let angle: f64 = match seq {
            0..90 => 0.,
            90..135 => 0.07,
            135..270 => 0.65,
            _ => -0.25 - ((seq - 270) as f64 / 30.).min(1.) * 0.55,
        };
        let (sin, cos) = angle.sin_cos();
        let [sx, _, sz] = arm.shoulder.meters.map(f64::from);
        for joint in [&mut arm.elbow, &mut arm.wrist] {
            let [x, y, z] = joint.meters.map(f64::from);
            joint.meters = [
                (sx + (x - sx) * cos + (z - sz) * sin) as f32,
                y as f32,
                (sz - (x - sx) * sin + (z - sz) * cos) as f32,
            ];
        }
        result.push(tick);
    }
    let file = File::create_new(out).map_err(|e| e.to_string())?;
    let mut buf = BufWriter::new(file);
    serde_json::to_writer(&mut buf, &result).map_err(|e| e.to_string())?;
    buf.flush().map_err(|e| e.to_string())?;
    println!("Seed frame {}", seed.1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationary_and_interrupted_samples_have_no_crossing_time() {
        let samples = [(0.0, Some(0.0)), (0.1, None), (0.2, Some(1.0))];
        assert_eq!(crossings(&samples, 0.0, 1.0, [0.0, 1.0]), [None; 3]);
        assert_eq!(crossings(&samples, 1.0, 1.0, [0.0, 1.0]), [None; 3]);
        let moving = [(0.0, Some(0.0)), (1.0, Some(1.0))];
        assert_eq!(
            crossings(&moving, 0.0, 1.0, [0.0, 2.0]),
            [Some(0.1), Some(0.5), Some(0.9)]
        );
    }
}
