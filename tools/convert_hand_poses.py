"""Convert the user-supplied CC0 UmebocDC archive into Rust hand-local poses.

Usage: python tools/convert_hand_poses.py path/to/UmebocDC_Hand.211010.zip
Only the converted angle table is saved; no Unity assets are extracted.
These are adapted poses, not a reproduction of Unity's avatar muscle solver.
See docs/adr/2026-10-03-hand-pose-library.md for the coordinate contract.
"""

import argparse
import hashlib
import io
import math
from pathlib import Path
import re
import tarfile
import zipfile


CLIPS = [
    ("Defo", "relaxed"), ("open", "open"), ("open2", "soft_open"),
    ("fist", "fist"), ("fist2", "soft_fist"), ("thumsup", "thumbs_up"),
    ("point", "point"), ("peace", "peace"), ("Kitsune", "fox"),
    ("rock'n'roll", "horns"), ("Tell", "shaka"), ("gun", "finger_gun"),
    ("Num3", "three"), ("Num4", "four"), ("Heart", "finger_heart"),
    ("Gao", "claw"), ("Agokui", "chin_support"), ("pose", "elegant"),
    ("pose2", "elegant_straight"), ("pose3", "gentle_open"),
]
FINGERS = ["Index", "Middle", "Ring", "Little"]
OPEN = [0, 0, 0]
FOLDED = [85, 100, 60]
# Additional authored poses in degrees, using the same MCP/PIP/DIP freedoms.
# Each tuple is (name, four finger curls, spreads, thumb MCP/IP, thumb spread).
ADDITIONAL = [
    ("ring_only", [FOLDED, FOLDED, OPEN, FOLDED], [0, 0, 0, 0], [40, 60], 35),
    ("little_only", [FOLDED, FOLDED, FOLDED, OPEN], [0, 0, 0, -12], [40, 60], 35),
    ("ok", [[45, 80, 35], OPEN, OPEN, OPEN], [10, 0, -8, -16], [35, 40], 28),
    ("crossed_fingers", [[12, 8, 0], [5, 5, 0], FOLDED, FOLDED], [-18, 15, 0, 0], [40, 55], 35),
    ("vulcan", [OPEN] * 4, [14, 14, -14, -14], [5, 0], 65),
    ("half_heart", [[35, 55, 25], [40, 60, 30], [45, 65, 35], [50, 70, 40]], [5, 0, -5, -10], [10, 5], 70),
    ("three_thumb_index_middle", [OPEN, OPEN, FOLDED, FOLDED], [12, -10, 0, 0], [0, 0], 65),
    ("i_love_you", [OPEN, FOLDED, FOLDED, OPEN], [12, 0, 0, -15], [0, 0], 70),
    ("four_with_thumb", [OPEN, OPEN, OPEN, FOLDED], [12, 0, -12, 0], [0, 0], 65),
    ("open_together", [OPEN] * 4, [0, 0, 0, 0], [5, 0], 25),
    ("two_together", [OPEN, OPEN, FOLDED, FOLDED], [0, 0, 0, 0], [40, 55], 35),
    ("pinch", [[30, 60, 25], [25, 30, 15], [30, 35, 20], [35, 40, 25]], [8, 0, -5, -10], [30, 35], 28),
]


def stretch(value, maximum):
    # +1 is extended, -1 is folded. Source overshoot is not hyperextension.
    return (1 - max(-1, min(1, value))) * 0.5 * maximum


def read_clips(archive):
    with zipfile.ZipFile(archive) as outer:
        package = next(n for n in outer.namelist() if n.endswith(".unitypackage"))
        with tarfile.open(fileobj=io.BytesIO(outer.read(package)), mode="r:gz") as inner:
            result = {}
            for member in inner.getmembers():
                if not member.name.endswith("/pathname"):
                    continue
                path = inner.extractfile(member).read().decode("utf-8")
                if "/Gesture Anim/" not in path or not path.endswith(".anim"):
                    continue
                asset = member.name.rsplit("/", 1)[0] + "/asset"
                text = inner.extractfile(asset).read().decode("utf-8")
                curves = text.split("  m_FloatCurves:")[1].split("  m_PPtrCurves:")[0]
                # This version contains one constant key per muscle curve.
                values = {}
                for block in curves.split("  - curve:")[1:]:
                    keys = re.findall(r"        value: ([^\n]+)", block)
                    attribute = re.search(r"    attribute: ([^\n]+)", block)
                    if len(keys) != 1 or attribute is None:
                        raise ValueError(f"Expected a constant muscle curve in {path}")
                    values[attribute[1]] = float(keys[0])
                result[Path(path).stem.removesuffix("_30ko")] = values
            return result


def convert(clip):
    def muscle(finger, channel):
        return clip[f"LeftHand.{finger}.{channel}"]

    curls = [[stretch(muscle(f, f"{joint} Stretched"), maximum)
              for joint, maximum in enumerate([90, 100, 60], 1)] for f in FINGERS]
    spreads = [max(-20, min(20, muscle(f, "Spread") * sign * 20))
               for f, sign in zip(FINGERS, [1, 1, -1, -1])]
    # Unity's thumb 1 is the CMC. The app intentionally leaves that bone at rest.
    thumb = [stretch(muscle("Thumb", "2 Stretched"), 50),
             stretch(muscle("Thumb", "3 Stretched"), 70)]
    thumb_spread = max(20, min(75, 55 + muscle("Thumb", "Spread") * 20))
    return curls, spreads, thumb, thumb_spread


def radian(value):
    for degrees, constant in [(90, "FRAC_PI_2"), (60, "FRAC_PI_3"),
                              (45, "FRAC_PI_4"), (30, "FRAC_PI_6"), (22.5, "FRAC_PI_8")]:
        if math.isclose(value, degrees, abs_tol=1e-6):
            return constant
    literal = f"{math.radians(value):.6f}".rstrip("0").rstrip(".")
    return literal if "." in literal else literal + ".0"


def radians(values):
    return "[" + ", ".join(radian(x) for x in values) + "]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    clips = read_clips(args.archive)
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    lines = [
        "// Generated by tools/convert_hand_poses.py; angles are radians.",
        "// UmebocDC_Hand.211010 (CC0), adapted from the left-hand muscle curves.",
        f"// Source archive SHA-256: {digest}",
        "// No Unity clips, models, textures or archive are shipped.",
        "use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8};",
        "use vtuber_core::arm_tracking::HandFingerPose;", "",
        "pub(super) const POSES: &[(&str, HandFingerPose)] = &[",
    ]
    poses = [(name, *convert(clips[clip])) for clip, name in CLIPS] + ADDITIONAL
    for name, fingers, spread, thumb, thumb_spread in poses:
        lines += [f'    ("{name}", HandFingerPose {{',
                  "        fingers: [" + ", ".join(radians(f) for f in fingers) + "],",
                  f"        spread: {radians(spread)},",
                  f"        thumb: {radians(thumb)},",
                  f"        thumb_spread: {radian(thumb_spread)},",
                  "    }),"]
    lines += ["];", ""]
    output = Path(__file__).resolve().parents[1] / "crates/vtuber-tracking/src/hand_poses/catalog.rs"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Converted {len(CLIPS)} CC0 poses + {len(ADDITIONAL)} authored poses: {output}")


if __name__ == "__main__":
    main()
