# ADR-001: 顔推論runtimeとmodel artifact

Status: Superseded by ADR-009 for production face tracking; retained as historical evidence
Date: 2026-08-04 (updated 2026-08-05)

> 2026-08-11のM1-08-015 rewriteにより、このADRのpure-Rust
> `tract-onnx`/PeppaPig/planar-pose production decisionはsupersededとなった。
> この文書は、旧実装の選定過程、再現可能なblocker、artifact provenanceを
> Git historyとともに保存する。現行production decisionはADR-009である。

## Context

本アプリはWindowsとmacOSで同じ顔追跡処理を実行し、配布時の推論runtimeをRustだけで構成する。顔追跡はmodel fileだけでは成立せず、detector、ROI、landmark、必要に応じたblendshape、前処理、後処理、tensor契約を一体として固定する必要がある。

TFLiteまたはONNXを読めることと、採用候補modelが実際に動くことは別である。unsupported operator、dynamic shape、quantization、custom postprocessのいずれかで失敗し得るため、runtime名を先に確定して実装を進めない。

## Failure baseline retained for the rewrite

The recorded Windows C922 guided runs established a correctness failure in the
old calibration and planar-pose path, rather than a user-behaviour problem:

- default calibration required 30 accepted observations within 5 seconds;
- measured landmark rates were approximately 0.293 Hz, 0.347 Hz, and 0.074 Hz,
  so the required sample count could not complete inside the timeout;
- recorded pose ranges reached yaw `[-5.145, 2.186]` rad, pitch
  `[-2.125, 4.209]` rad, roll `[-2.605, 1.199]` rad, with a later pitch maximum
  of `8.091` rad;
- the PeppaPig manifest declared ImageNet normalization
  `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`, while the upstream
  implementation and PINTO demo use RGB divided by `255.0` with zero mean and
  unit standard deviation;
- the old 98-point expression mapping used MediaPipe 478-point indices
  `133, 160, 263, 291, 362, 382, 388`, which are out of range for a 98-point
  result and therefore cannot be a valid production blink/mouth mapping.

These measurements supersede the old assumption that a stricter stillness gate
or a wider rejection threshold would repair the pipeline. They are retained as
the baseline against which the MediaPipe rewrite is evaluated.

## Historical fixed decision (superseded)

- production runtimeはpure Rustとする。
- Python subprocess、MediaPipe native runtime、TensorFlow Lite C API、ONNX Runtime、OpenCV DNNをproduction pathへ含めない。
- WindowsとmacOSで同一artifact、同一前後処理、同一golden contractを使う。
- TFLite backendとONNX backendをproduction buildで同時に有効化しない。
- compatibility failure時にnative runtimeへ無断で切り替えない。

The current user-authorized exception is recorded in ADR-009: official
MediaPipe Tasks 0.10.35 through the pinned audited `mediapipe-rs` revision.
This is not a general authorization for arbitrary native inference runtimes.

## Candidate order

1. `tract-tflite = 0.23.4` (ADR原文: 0.23.0; crates.ioに 0.23.0 が存在しないため 0.23.4 に修正)
2. `tract-onnx = 0.23.4`

## Resolution

- `tract-tflite` 0.23.4 で MediaPipe Face Landmarker `.task` 内の TFLite モデル (`face_detector.tflite`, `face_landmarks_detector.tflite`, `face_blendshapes.tflite`) および従来の `face_landmark.tflite` を読み込もうとすると、`F16 ADD` operator (builtin_code ADD, version 2) が unsupported として失敗する。
- 第二候補 `tract-onnx` 0.23.4 では、PINTO_model_zoo 経由で提供されている `peppapig_student_1x3x256x256.onnx` (Peppa_Pig_Face_Landmark, Apache-2.0) が正常に load / optimize でき、入力 `[1,3,256,256]` F32、出力 `[1,98,3]` F32 を確認した。
- 本 ADR の採用 runtime は `tract-onnx = 0.23.4` とし、production feature は `onnx` とする。TFLite パスは fallback として Cargo feature `tflite` にはせず、G0-05 時点では非採用と記録する。

## TFLite blocker record

- operator: `ADD` (builtin_code 6, version 2) with F16 tensor
- affected artifacts:
  - `face_landmarker.task` → extracted `face_detector.tflite`, `face_landmarks_detector.tflite`, `face_blendshapes.tflite`
  - `face_landmark.tflite` (legacy MediaPipe Face Mesh, float32)
- SHA-256: see `assets/models/manifest.toml`
- reproduction: `cargo test -p vtuber-inference --features onnx probe_face_detector -- --ignored --nocapture` (TFLite load fails)
- error pattern: `Unsupported: OperatorCode { deprecated_builtin_code: 6, custom_code: None, version: 2, builtin_code: ADD }, inputs: [16,3,3,3,F16 ...]`

## ONNX model provenance

- model: Peppa_Pig_Face_Landmark student 256x256 ONNX
- upstream: https://github.com/610265158/Peppa_Pig_Face_Landmark (Apache-2.0)
- converted/distributed by: PINTO0309/PINTO_model_zoo (conversion scripts MIT; model follows upstream license)
- download: https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/436_Peppa_Pig_Face_Landmark/resources.tar.gz
- local file: `assets/models/peppapig_student_1x3x256x256.onnx`
- SHA-256: `73EDF90954F05EBEF4639E7FA8620C5F83CCA09D2476DE66AB100F26C2B25E7A`
- input: `[1,3,256,256]` F32 NCHW, ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- output: `[1,98,3]` F32, 98 facial landmarks with confidence/visibility in third channel
- validation: `cargo test -p vtuber-inference --features onnx probe_peppapig_student_256 -- --ignored --nocapture` passes on Windows x86_64

第二候補へ進むのは、第一候補のblockerをoperator名、model SHA-256、入力shape、再現commandとともに記録した場合に限る。ONNX変換を行う場合はconverterのversionとcommandを固定し、original artifactとのgolden比較を必須とする。今回採用した ONNX artifact は PINTO_model_zoo が既に変換済みのものであり、upstream と同じ Apache-2.0 license の下で再配布可能である。

## G0-05 acceptance evidence

- detector／landmark／optional blendshape artifactの一次取得元
- licenseとredistribution条件
- SHA-256
- input dtype、shape、layout、normalization
- output indexまたはname、shape、意味
- ROI変換とlandmark座標の定義
- operator inventory
- fixed inputに対するgolden outputとtolerance
- WindowsとmacOSのload／optimize／run結果
- release buildのp50／p95 inference time
- model manifestとlicense file

## Consequences

model/runtime選定はblocking gateとなる。Gateを通るまで、UI、tracking core、VRM adapterを特定modelのtensor indexへ直接結合しない。`FaceInference`と`LandmarkSchemaId`を境界とし、model固有indexはschema adapterへ閉じ込める。

## M1-08-013 resolution: 2D landmark pose adapter

The production choice remains the manifest-listed `peppapig-98` ONNX model.
Its output is image-space `(x, y, confidence)` data, not a metric 3D point
cloud. The application therefore does not pass zero-filled `z` values to the
3D Kabsch solver.

The selected option is the second option from the task gate: a pure-Rust
orthographic solver in `vtuber-tracking::pose::planar`. It fits yaw, pitch,
roll, scale, and translation against a small synthetic, license-safe canonical
face template. The adapter records representative WFLW-98 output indices
`[16, 37, 46, 52, 63, 71, 76, 82]` in `assets/models/manifest.toml` and uses
the design convention of positive image-right yaw, chin-up pitch, and clockwise
image roll.

This resolves the 2D/3D type mismatch and has deterministic synthetic sign
tests. The manifest-tracked UltraFace RFB-320 artifact, pure-Rust detector
decode/NMS, primary-face selection, square crop, and same-worker composite
runtime are now implemented and covered by fixed-input golden/replay tests. A
Windows MSMF diagnostic command is also available. A still-image probe on a
user-supplied 800x533 JPEG detected one face at confidence 0.9999, confirming
the detector artifact, RGBA8 preprocessing, and output decoder. The Windows
vertical gate initially returned `NO_FACE` despite the captured C922 frame
containing a face. The cause was landmark-model visibility values above the
engine contract; the ONNX landmark decoder now clamps that channel to `[0,1]`,
matching the generic decoder contract. A subsequent 20-second C922 run
produced 6 face observations, 98 finite landmarks, and 5 finite planar poses
with no stage error. The full M1-08-013 camera protocol was then exercised
with the guided CUI on the same C922: 4,944 frames, 50 face observations, 19
expected no-face observations during out-of-frame phases, 98 finite
landmarks, 44 finite planar poses, detector/landmark rates of 0.387/0.347 Hz,
and no stage error. The protocol included five loss/return cycles, directional
movement, an edge phase, and three capture Stop/Start cycles. The parent
camera gate is accepted for this Windows environment; GUI/VRM and long-run
acceptance remain separate gates.

## M1-08-013-002 exact UltraFace probe result

The repair leaf fixes the detector candidate to ultraface-rfb-320 and the
runtime to tract-onnx = 0.23.4. The exact artifact identity is
version-RFB-320.onnx, 1,270,727 bytes, SHA-256
34CD7E60AEFF28744C657DE7A3DC64E872D506741DE66987F3426F2B79F88017.
The current ONNX Model Zoo mirror is
https://huggingface.co/onnxmodelzoo/version-RFB-320/resolve/main/version-RFB-320.onnx.

On 2026-08-11 the exact local artifact passed the tract-onnx 0.23.4 probe:
load, input fact validation, operator inventory, optimize, runnable
construction, and execution all passed. The loaded input is F32
[1,3,240,320]. Outputs are ordered as scores F32 [1,4420,2] and boxes F32
[1,4420,4]. Both zero and mean-normalized runs produced finite values; the
reported element counts are 8,840 and 17,680 respectively. The stable
452-node source-graph inventory is recorded in
assets/models/operator-inventory-ultraface.txt. The model-with-test-data
artifact was not required for this exact contract probe.
