// bindgen entry point. Public MediaPipe Tasks C headers only — the *_converter.h
// headers are C++ internals that pull in absl and Eigen.
//
// interactive_segmenter_legacy.h is included solely so that
// MpInteractiveSegmenterLegacyCreate gets bound: its presence/absence is how
// loader.rs distinguishes the v0.10.35 ABI from the post-rename one. See
// src/sys/compat.rs.

#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/base_options.h"

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/matrix.h"
#include "mediapipe/tasks/c/components/containers/rect.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"

#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"

#include "mediapipe/tasks/c/vision/face_detector/face_detector.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"

#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result.h"

#include "mediapipe/tasks/c/vision/interactive_segmenter_legacy/interactive_segmenter_legacy.h"
