"""Convert the user-supplied CC0 UmebocDC archive into Rust hand-local poses.

Usage: python tools/convert_hand_poses.py path/to/UmebocDC_Hand.211010.zip
Only the converted angle table is saved; no Unity assets are extracted.
These are adapted poses, not a reproduction of Unity's avatar muscle solver.
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
# Each tuple is (name, four finger curls, spreads, thumb spread).
ADDITIONAL = [
    ("ring_only", [FOLDED, FOLDED, OPEN, FOLDED], [0, 0, 0, 0], 35),
    ("little_only", [FOLDED, FOLDED, FOLDED, OPEN], [0, 0, 0, -12], 35),
    ("ok", [[32, 75, 21], OPEN, OPEN, OPEN], [20, 0, -8, -16], 28),
    ("crossed_fingers", [[12, 8, 0], [5, 5, 0], FOLDED, FOLDED], [-18, 15, 0, 0], 35),
    ("vulcan", [OPEN] * 4, [14, 14, -14, -14], 65),
    ("half_heart", [[35, 55, 25], [40, 60, 30], [45, 65, 35], [50, 70, 40]], [5, 0, -5, -10], 70),
    ("three_thumb_index_middle", [OPEN, OPEN, FOLDED, FOLDED], [12, -10, 0, 0], 65),
    ("i_love_you", [OPEN, FOLDED, FOLDED, OPEN], [12, 0, 0, -15], 70),
    ("four_with_thumb", [OPEN, OPEN, OPEN, FOLDED], [12, 0, -12, 0], 65),
    ("open_together", [OPEN] * 4, [0, 0, 0, 0], 25),
    ("two_together", [OPEN, OPEN, FOLDED, FOLDED], [0, 0, 0, 0], 35),
    ("pinch", [[32, 75, 21], [25, 30, 15], [30, 35, 20], [35, 40, 25]], [20, 0, -5, -10], 28),
]

# Artist-authored thumb coordinates for the anatomical VRM adapter, in degrees:
# (MCP/IP flexion, CMC flexion/abduction). Unity's normalized thumb muscles do
# not specify this model's oblique joint angles. In particular, a folded thumb
# must oppose the fingers, rather than inherit positive CMC extension from a
# VRChat clip. The spread feature remains available for pose recognition.
THUMBS = {
    "relaxed": ([25, 17.5], [0, 0]),
    "open": ([0, 0], [0, 7]),
    "soft_open": ([12.5, 0], [-21, 0]),
    "fist": ([37.5, 70], [0, 37]),
    "soft_fist": ([30, 30], [-20, 25]),
    "thumbs_up": ([12.5, 0], [-30, 0]),
    "point": ([35, 50], [-25, 25]),
    "peace": ([35, 55], [-30, 25]),
    "fox": ([25, 20], [-15, 35]),
    "horns": ([35, 55], [-30, 25]),
    "shaka": ([0, 0], [0, 0]),
    "finger_gun": ([5, 0], [-30, 0]),
    "three": ([40, 35], [-35, 30]),
    "four": ([37.5, 52.5], [-25, 25]),
    "finger_heart": ([25, 0], [-15, 30]),
    "claw": ([5, 52.5], [0, 37]),
    "chin_support": ([20, 0], [-21, 0]),
    "elegant": ([25, 0], [0, 18]),
    "elegant_straight": ([12.5, 0], [0, 37]),
    "gentle_open": ([37.5, 0], [-13, -18]),
    "ring_only": ([40, 60], [-20, 20]),
    "little_only": ([40, 60], [-20, 20]),
    "ok": ([15, 30], [-8, 44]),
    "crossed_fingers": ([40, 55], [-20, 20]),
    "vulcan": ([5, 0], [10, -10]),
    "half_heart": ([10, 5], [-10, -10]),
    "three_thumb_index_middle": ([0, 0], [15, -5]),
    "i_love_you": ([0, 0], [15, -10]),
    "four_with_thumb": ([0, 0], [15, -5]),
    "open_together": ([5, 0], [0, 10]),
    "two_together": ([40, 55], [-20, 20]),
    "pinch": ([15, 30], [-8, 44]),
}


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


def convert(clip, name):
    def muscle(finger, channel):
        return clip[f"LeftHand.{finger}.{channel}"]

    curls = [[stretch(muscle(f, f"{joint} Stretched"), maximum)
              for joint, maximum in enumerate([90, 100, 60], 1)] for f in FINGERS]
    spreads = [max(-20, min(20, muscle(f, "Spread") * sign * 20))
               for f, sign in zip(FINGERS, [1, 1, -1, -1])]
    thumb, cmc = THUMBS[name]
    thumb_spread = max(20, min(75, 55 + muscle("Thumb", "Spread") * 20))
    return curls, spreads, thumb, thumb_spread, cmc


def radian(value):
    if value < 0:
        return "-" + radian(-value)
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
        "// Thumb joint coordinates are artist-authored for the anatomical VRM axes.",
        f"// Source archive SHA-256: {digest}",
        "// No Unity clips, models, textures or archive are shipped.",
        "use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8};",
        "use vtuber_core::arm_tracking::HandFingerPose;", "",
        "pub(super) const POSES: &[(&str, HandFingerPose)] = &[",
    ]
    poses = [(name, *convert(clips[clip], name)) for clip, name in CLIPS]
    poses += [(name, fingers, spread, THUMBS[name][0], thumb_spread, THUMBS[name][1])
              for name, fingers, spread, thumb_spread in ADDITIONAL]
    for name, fingers, spread, thumb, thumb_spread, cmc in poses:
        lines += [f'    ("{name}", HandFingerPose {{',
                  "        fingers: [" + ", ".join(radians(f) for f in fingers) + "],",
                  f"        spread: {radians(spread)},",
                  f"        thumb: {radians(thumb)},",
                  f"        thumb_cmc: {radians(cmc)},",
                  f"        thumb_spread: {radian(thumb_spread)},",
                  "    }),"]
    lines += ["];", ""]
    output = Path(__file__).resolve().parents[1] / "crates/vtuber-tracking/src/hand_poses/catalog.rs"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Converted {len(CLIPS)} CC0 poses + {len(ADDITIONAL)} authored poses: {output}")


if __name__ == "__main__":
    main()
