/*
 * Copyright 2022 l1npengtul <l1npengtul@protonmail.com> / The Nokhwa Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#![deny(clippy::pedantic)]
#![warn(clippy::all)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::too_many_lines)]

//! # nokhwa-bindings-windows
//! This crate is the `MediaFoundation` bindings for the `nokhwa` crate.
//!
//! It is not meant for general consumption. If you are looking for a Windows camera capture crate, consider using `nokhwa` with feature `input-msmf`.
//!
//! No support or API stability will be given. Subject to change at any time.

#[cfg(all(windows, not(feature = "docs-only")))]
pub mod wmf {
    use nokhwa_core::error::NokhwaError;
    use nokhwa_core::types::{
        ApiBackend, CameraControl, CameraFormat, CameraIndex, CameraInfo, ControlValueDescription,
        ControlValueSetter, FrameFormat, KnownCameraControl, KnownCameraControlFlag, Resolution,
    };
    use std::ffi::c_void;
    use std::{borrow::Cow, cell::Cell, mem::MaybeUninit, slice::from_raw_parts, time::Duration};
    use windows::Win32::Media::DirectShow::{CameraControl_Flags_Auto, CameraControl_Flags_Manual};
    use windows::Win32::Media::MediaFoundation::{
        MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
    };
    use windows::{
        core::{Interface, GUID, PWSTR},
        Win32::{
            Foundation::RPC_E_CHANGED_MODE,
            Media::{
                DirectShow::{
                    CameraControl_Exposure, CameraControl_Focus, CameraControl_Iris,
                    CameraControl_Pan, CameraControl_Tilt, CameraControl_Zoom, IAMCameraControl,
                    IAMVideoProcAmp, VideoProcAmp_BacklightCompensation, VideoProcAmp_Brightness,
                    VideoProcAmp_ColorEnable, VideoProcAmp_Contrast, VideoProcAmp_Gain,
                    VideoProcAmp_Gamma, VideoProcAmp_Hue, VideoProcAmp_Saturation,
                    VideoProcAmp_Sharpness, VideoProcAmp_WhiteBalance,
                },
                KernelStreaming::GUID_NULL,
                MediaFoundation::{
                    IMFActivate, IMFAttributes, IMFMediaSource, IMFSample, IMFSourceReader,
                    MFCreateAttributes, MFCreateSourceReaderFromMediaSource, MFEnumDeviceSources,
                    MFShutdown, MFStartup, MFSTARTUP_NOSOCKET,
                    MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
                    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, MF_MT_FRAME_RATE,
                    MF_MT_FRAME_RATE_RANGE_MAX, MF_MT_FRAME_RATE_RANGE_MIN, MF_MT_FRAME_SIZE,
                    MF_MT_SUBTYPE, MF_READWRITE_DISABLE_CONVERTERS, MF_VERSION,
                },
            },
            System::Com::{CoInitializeEx, CoTaskMemFree, CoUninitialize, COINIT},
        },
    };

    // See: https://stackoverflow.com/questions/80160/what-does-coinit-speed-over-memory-do
    const CO_INIT_APARTMENT_THREADED: COINIT = COINIT(0x2);
    const CO_INIT_DISABLE_OLE1DDE: COINIT = COINIT(0x4);

    // See: https://gix.github.io/media-types/#major-types
    const MF_VIDEO_FORMAT_YUY2: GUID = GUID::from_values(
        0x3259_5559,
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    );
    const MF_VIDEO_FORMAT_MJPEG: GUID = GUID::from_values(
        0x4750_4A4D,
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    );
    const MF_VIDEO_FORMAT_GRAY: GUID = GUID::from_values(
        0x3030_3859,
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    );
    const MF_VIDEO_FORMAT_NV12: GUID = GUID::from_values(
        0x3231_564E,
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    );
    const MF_VIDEO_FORMAT_RGB24: GUID = GUID::from_values(
        0x0000_0014,
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    );

    const MEDIA_FOUNDATION_FIRST_VIDEO_STREAM: u32 = 0xFFFF_FFFC;
    const MF_SOURCE_READER_MEDIASOURCE: u32 = 0xFFFF_FFFF;

    // const CAM_CTRL_AUTO: i32 = 0x0001;
    // const CAM_CTRL_MANUAL: i32 = 0x0002;

    // macro_rules! define_controls {
    //     ( $( ($key:expr => ($property:ident, $min:ident, $max:ident, $step:ident, $default:ident, $flag:ident)) )* ) => {
    //         $(
    //         $key => {
    //             if let Err(why) = unsafe {
    //                     video_proc_amp.GetRange(
    //                         $property.0,
    //                         &mut $min,
    //                         &mut $max,
    //                         &mut $step,
    //                         &mut $default,
    //                         &mut $flag,
    //                     )
    //                 } {
    //                     return Err(NokhwaError::GetPropertyError {
    //                         property: stringify!($key).to_string(),
    //                         error: why.to_string()
    //                     });
    //                 }
    //         }
    //         )*
    //     };
    //     ( $( ($key:expr : ($property:ident, $value:ident, $flag:ident)) )* ) => {
    //         $(
    //         $key => {
    //             if let Err(why) = unsafe {
    //                 video_proc_amp.Get($property.0, &mut $value, &mut $flag)
    //                 } {
    //                     return Err(NokhwaError::GetPropertyError {
    //                         property: stringify!($key).to_string(),
    //                         error: why.to_string()
    //                     });
    //                 }
    //         }
    //         )*
    //     };
    // }

    fn guid_to_frameformat(guid: GUID) -> Option<FrameFormat> {
        match guid {
            MF_VIDEO_FORMAT_NV12 => Some(FrameFormat::NV12),
            MF_VIDEO_FORMAT_RGB24 => Some(FrameFormat::RAWBGR),
            MF_VIDEO_FORMAT_GRAY => Some(FrameFormat::GRAY),
            MF_VIDEO_FORMAT_YUY2 => Some(FrameFormat::YUYV),
            MF_VIDEO_FORMAT_MJPEG => Some(FrameFormat::MJPEG),
            _ => None,
        }
    }

    // COM belongs to the calling thread. MFStartup/MFShutdown are balanced for
    // each enumeration and device, so dropping a camera cannot shut down MF
    // while another thread is enumerating. Rc prevents cross-thread teardown.
    struct MfRuntime {
        uninitialize_com: bool,
        _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
    }

    impl MfRuntime {
        fn new() -> Result<Self, NokhwaError> {
            // Reuse an existing apartment without changing its model or taking
            // ownership of its COM initialization (e.g. the application's UI).
            let result = unsafe {
                CoInitializeEx(None, CO_INIT_APARTMENT_THREADED | CO_INIT_DISABLE_OLE1DDE)
            };
            if result.is_err() && result != RPC_E_CHANGED_MODE {
                return Err(NokhwaError::InitializeError {
                    backend: ApiBackend::MediaFoundation,
                    error: result.to_string(),
                });
            }
            let uninitialize_com = result.is_ok();
            if let Err(why) = unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) } {
                if uninitialize_com {
                    unsafe { CoUninitialize() };
                }
                return Err(NokhwaError::InitializeError {
                    backend: ApiBackend::MediaFoundation,
                    error: why.to_string(),
                });
            }
            Ok(Self {
                uninitialize_com,
                _thread_bound: std::marker::PhantomData,
            })
        }
    }

    impl Drop for MfRuntime {
        fn drop(&mut self) {
            // All owned COM interfaces must be released before this guard.
            unsafe {
                let _ = MFShutdown();
                if self.uninitialize_com {
                    CoUninitialize();
                }
            }
        }
    }

    // Every activated capture source must be shut down, including when source
    // reader construction fails. Release the reader before dropping this owner.
    struct CaptureSource(IMFMediaSource);

    impl Drop for CaptureSource {
        fn drop(&mut self) {
            unsafe {
                let _ = self.0.Shutdown();
            }
        }
    }

    fn query_activate_pointers() -> Result<Vec<IMFActivate>, NokhwaError> {
        let mut attributes: Option<IMFAttributes> = None;
        if let Err(why) = unsafe { MFCreateAttributes(&mut attributes, 1) } {
            return Err(NokhwaError::GetPropertyError {
                property: "IMFAttributes".to_string(),
                error: why.to_string(),
            });
        }

        let attributes = match attributes {
            Some(attr) => {
                if let Err(why) = unsafe {
                    attr.SetGUID(
                        &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                        &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
                    )
                } {
                    return Err(NokhwaError::SetPropertyError {
                        property: "GUID MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE".to_string(),
                        value: "MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID".to_string(),
                        error: why.to_string(),
                    });
                }
                attr
            }
            None => {
                return Err(NokhwaError::SetPropertyError {
                    property: "GUID MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE".to_string(),
                    value: "MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID".to_string(),
                    error: "Call to IMFAttributes::SetGUID failed - IMFAttributes is None"
                        .to_string(),
                });
            }
        };

        let mut count: u32 = 0;
        let mut pointers: *mut Option<IMFActivate> = std::ptr::null_mut();
        if let Err(why) = unsafe { MFEnumDeviceSources(&attributes, &mut pointers, &mut count) } {
            return Err(NokhwaError::StructureError {
                structure: "MFEnumDeviceSources".to_string(),
                error: why.to_string(),
            });
        }

        // Transfer each returned interface reference before freeing the COM
        // array. Cloning without releasing the originals leaks activations.
        let mut device_list = Vec::new();
        if !pointers.is_null() {
            for index in 0..count as usize {
                if let Some(activate) = unsafe { pointers.add(index).read() } {
                    device_list.push(activate);
                }
            }
            unsafe { CoTaskMemFree(Some(pointers.cast())) };
        }

        Ok(device_list)
    }

    fn activate_to_descriptors(
        index: CameraIndex,
        imf_activate: &IMFActivate,
    ) -> Result<CameraInfo, NokhwaError> {
        let name = allocated_string(imf_activate, &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)?;
        let symlink = allocated_string(
            imf_activate,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
        )?;

        Ok(CameraInfo::new(
            &name,
            "MediaFoundation Camera",
            &symlink,
            index,
        ))
    }

    fn allocated_string(activate: &IMFActivate, key: &GUID) -> Result<String, NokhwaError> {
        let mut value = PWSTR::null();
        let mut length = 0;
        unsafe { activate.GetAllocatedString(key, &mut value, &mut length) }.map_err(|error| {
            NokhwaError::GetPropertyError {
                property: format!("{key:?}"),
                error: error.to_string(),
            }
        })?;
        let result = if value.is_null() {
            Err(NokhwaError::StructureError {
                structure: "IMFActivate string".to_string(),
                error: "null string".to_string(),
            })
        } else {
            unsafe { value.to_string() }.map_err(|error| NokhwaError::StructureError {
                structure: "IMFActivate string".to_string(),
                error: error.to_string(),
            })
        };
        unsafe { CoTaskMemFree(Some(value.0.cast())) };
        result
    }

    pub fn query_media_foundation_descriptors() -> Result<Vec<CameraInfo>, NokhwaError> {
        let _runtime = MfRuntime::new()?;
        let mut device_list = vec![];

        for (index, activate_ptr) in query_activate_pointers()?.into_iter().enumerate() {
            device_list.push(activate_to_descriptors(
                CameraIndex::Index(index as u32),
                &activate_ptr,
            )?);
        }
        Ok(device_list)
    }

    #[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Eq)]
    enum MFControlId {
        ProcAmpBoolean(i32),
        ProcAmpRange(i32),
        CCValue(i32),
        CCRange(i32),
    }

    #[allow(clippy::cast_sign_loss)]
    fn kcc_to_i32(kcc: KnownCameraControl) -> Option<MFControlId> {
        let control_id = match kcc {
            KnownCameraControl::Brightness => MFControlId::ProcAmpRange(VideoProcAmp_Brightness.0),
            KnownCameraControl::Contrast => MFControlId::ProcAmpRange(VideoProcAmp_Contrast.0),
            KnownCameraControl::Hue => MFControlId::ProcAmpRange(VideoProcAmp_Hue.0),
            KnownCameraControl::Saturation => MFControlId::ProcAmpRange(VideoProcAmp_Saturation.0),
            KnownCameraControl::Sharpness => MFControlId::ProcAmpRange(VideoProcAmp_Sharpness.0),
            KnownCameraControl::Gamma => MFControlId::ProcAmpRange(VideoProcAmp_Gamma.0),
            KnownCameraControl::WhiteBalance => {
                MFControlId::ProcAmpRange(VideoProcAmp_WhiteBalance.0)
            }
            KnownCameraControl::BacklightComp => {
                MFControlId::ProcAmpBoolean(VideoProcAmp_BacklightCompensation.0)
            }
            KnownCameraControl::Gain => MFControlId::ProcAmpRange(VideoProcAmp_Gain.0),
            KnownCameraControl::Pan => MFControlId::CCRange(CameraControl_Pan.0),
            KnownCameraControl::Tilt => MFControlId::CCRange(CameraControl_Tilt.0),
            KnownCameraControl::Zoom => MFControlId::CCRange(CameraControl_Zoom.0),
            KnownCameraControl::Exposure => MFControlId::CCValue(CameraControl_Exposure.0),
            KnownCameraControl::Iris => MFControlId::CCValue(CameraControl_Iris.0),
            KnownCameraControl::Focus => MFControlId::CCValue(CameraControl_Focus.0),
            KnownCameraControl::Other(o) => {
                if o == VideoProcAmp_ColorEnable.0 as u128 {
                    MFControlId::ProcAmpRange(o as i32)
                } else {
                    return None;
                }
            }
        };

        Some(control_id)
    }

    pub struct MediaFoundationDevice {
        is_open: Cell<bool>,
        device_specifier: CameraInfo,
        device_format: CameraFormat,
        source_reader: IMFSourceReader,
        _source: CaptureSource,
        _runtime: MfRuntime,
        /// Wallclock instant captured when the stream was started.
        /// MF sample timestamps are relative to stream start, so
        /// `stream_epoch + sample_time` gives us an absolute wallclock.
        stream_epoch: Option<Duration>,
    }

    impl MediaFoundationDevice {
        pub fn new(index: CameraIndex) -> Result<Self, NokhwaError> {
            let runtime = MfRuntime::new()?;
            // Match and activate within one enumeration. Resolving a symbolic
            // link to an index and then enumerating again races hot-plugging.
            let mut selected = None;
            for (position, activate) in query_activate_pointers()?.into_iter().enumerate() {
                let descriptor =
                    activate_to_descriptors(CameraIndex::Index(position as u32), &activate)?;
                let matches = match &index {
                    CameraIndex::Index(wanted) => position == *wanted as usize,
                    CameraIndex::String(wanted) => descriptor.misc() == *wanted,
                };
                if matches {
                    selected = Some((activate, descriptor));
                    break;
                }
            }
            let (activate, device_descriptor) = selected.ok_or_else(|| {
                NokhwaError::OpenDeviceError(index.to_string(), "Not Found".to_string())
            })?;
            let media_source = CaptureSource(
                unsafe { activate.ActivateObject::<IMFMediaSource>() }.map_err(|error| {
                    NokhwaError::OpenDeviceError(index.to_string(), error.to_string())
                })?,
            );
            // Do not leave the activation object caching the source reference.
            // CaptureSource now owns shutdown, including on subsequent errors.
            unsafe { activate.DetachObject() }.map_err(|error| {
                NokhwaError::OpenDeviceError(index.to_string(), error.to_string())
            })?;
            drop(activate);
            let mut attributes = None;
            unsafe { MFCreateAttributes(&mut attributes, 1) }.map_err(|error| {
                NokhwaError::StructureError {
                    structure: "MFCreateAttributes".to_string(),
                    error: error.to_string(),
                }
            })?;
            let attributes = attributes.ok_or_else(|| NokhwaError::StructureError {
                structure: "MFCreateAttributes".to_string(),
                error: "No attributes".to_string(),
            })?;
            unsafe { attributes.SetUINT32(&MF_READWRITE_DISABLE_CONVERTERS, 1) }.map_err(
                |error| NokhwaError::SetPropertyError {
                    property: "MF_READWRITE_DISABLE_CONVERTERS".to_string(),
                    value: "1".to_string(),
                    error: error.to_string(),
                },
            )?;
            let source_reader =
                unsafe { MFCreateSourceReaderFromMediaSource(&media_source.0, &attributes) }
                    .map_err(|error| NokhwaError::StructureError {
                        structure: "MFCreateSourceReaderFromMediaSource".to_string(),
                        error: error.to_string(),
                    })?;
            Ok(Self {
                is_open: Cell::new(false),
                device_specifier: device_descriptor,
                device_format: CameraFormat::default(),
                source_reader,
                _source: media_source,
                _runtime: runtime,
                stream_epoch: None,
            })
        }

        pub fn index(&self) -> &CameraIndex {
            self.device_specifier.index()
        }

        pub fn name(&self) -> String {
            self.device_specifier.human_name()
        }

        pub fn symlink(&self) -> String {
            self.device_specifier.misc()
        }

        pub fn compatible_format_list(&mut self) -> Result<Vec<CameraFormat>, NokhwaError> {
            let mut camera_format_list = vec![];
            let mut index = 0;

            while let Ok(media_type) = unsafe {
                self.source_reader
                    .GetNativeMediaType(MEDIA_FOUNDATION_FIRST_VIDEO_STREAM, index)
            } {
                index += 1;
                let fourcc = match unsafe { media_type.GetGUID(&MF_MT_SUBTYPE) } {
                    Ok(fcc) => fcc,
                    Err(why) => {
                        return Err(NokhwaError::GetPropertyError {
                            property: "MF_MT_SUBTYPE".to_string(),
                            error: why.to_string(),
                        })
                    }
                };

                let (width, height) = match unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) } {
                    Ok(res_u64) => {
                        let width = (res_u64 >> 32) as u32;
                        let height = res_u64 as u32; // the cast will truncate the upper bits
                        (width, height)
                    }
                    Err(why) => {
                        return Err(NokhwaError::GetPropertyError {
                            property: "MF_MT_FRAME_SIZE".to_string(),
                            error: why.to_string(),
                        })
                    }
                };

                // MFRatio is represented as 2 u32s in memory. This means we can convert it to 2
                let framerate_list = {
                    let mut framerates = vec![0_u32; 3];
                    if let Ok(fraction_u64) =
                        unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE_RANGE_MAX) }
                    {
                        let mut numerator = (fraction_u64 >> 32) as u32;
                        let denominator = fraction_u64 as u32;
                        if denominator != 1 {
                            numerator = 0;
                        }
                        framerates.push(numerator);
                    };
                    if let Ok(fraction_u64) = unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE) } {
                        let mut numerator = (fraction_u64 >> 32) as u32;
                        let denominator = fraction_u64 as u32;
                        if denominator != 1 {
                            numerator = 0;
                        }
                        framerates.push(numerator);
                    };
                    if let Ok(fraction_u64) =
                        unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE_RANGE_MIN) }
                    {
                        let mut numerator = (fraction_u64 >> 32) as u32;
                        let denominator = fraction_u64 as u32;
                        if denominator != 1 {
                            numerator = 0;
                        }
                        framerates.push(numerator);
                    };
                    framerates
                };

                let frame_fmt = match guid_to_frameformat(fourcc) {
                    Some(fcc) => fcc,
                    None => continue,
                };

                for frame_rate in framerate_list {
                    if frame_rate != 0 {
                        camera_format_list.push(CameraFormat::new(
                            Resolution::new(width, height),
                            frame_fmt,
                            frame_rate,
                        ));
                    }
                }
            }
            Ok(camera_format_list)
        }

        pub fn control(&self, control: KnownCameraControl) -> Result<CameraControl, NokhwaError> {
            let camera_control = unsafe {
                let mut receiver: MaybeUninit<IAMCameraControl> = MaybeUninit::uninit();
                let ptr_receiver = receiver.as_mut_ptr();
                if let Err(why) = self.source_reader.GetServiceForStream(
                    MF_SOURCE_READER_MEDIASOURCE,
                    &GUID_NULL,
                    &IAMCameraControl::IID,
                    ptr_receiver
                        .cast::<IAMCameraControl>()
                        .cast::<*mut c_void>(),
                ) {
                    return Err(NokhwaError::SetPropertyError {
                        property: "MF_SOURCE_READER_MEDIASOURCE".to_string(),
                        value: "IAMCameraControl".to_string(),
                        error: why.to_string(),
                    });
                }
                receiver.assume_init()
            };
            let video_proc_amp = unsafe {
                let mut receiver: MaybeUninit<IAMVideoProcAmp> = MaybeUninit::uninit();
                let ptr_receiver = receiver.as_mut_ptr();
                if let Err(why) = self.source_reader.GetServiceForStream(
                    MF_SOURCE_READER_MEDIASOURCE,
                    &GUID_NULL,
                    &IAMVideoProcAmp::IID,
                    ptr_receiver.cast::<IAMVideoProcAmp>().cast::<*mut c_void>(),
                ) {
                    return Err(NokhwaError::SetPropertyError {
                        property: "MF_SOURCE_READER_MEDIASOURCE".to_string(),
                        value: "IAMVideoProcAmp".to_string(),
                        error: why.to_string(),
                    });
                }
                receiver.assume_init()
            };

            let mut min = 0;
            let mut max = 0;
            let mut step = 0;
            let mut default = 0;
            let mut value = 0;
            let mut flag = 0;

            let control_id = kcc_to_i32(control).ok_or(NokhwaError::SetPropertyError {
                property: "CameraControl".to_string(),
                value: control.to_string(),
                error: "Does not exist".to_string(),
            })?;

            let ctrl_value_set = match control_id {
                MFControlId::ProcAmpBoolean(id) => unsafe {
                    if let Err(why) = video_proc_amp.GetRange(
                        id,
                        &mut min,
                        &mut max,
                        &mut step,
                        &mut default,
                        &mut flag,
                    ) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Range", control_id, control),
                            error: why.to_string(),
                        });
                    }
                    if let Err(why) = video_proc_amp.Get(id, &mut value, &mut flag) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Value", control_id, control),
                            error: why.to_string(),
                        });
                    }

                    let boolval = value != 0;
                    let booldef = default != 0;
                    ControlValueDescription::Boolean {
                        value: boolval,
                        default: booldef,
                    }
                },
                MFControlId::ProcAmpRange(id) => unsafe {
                    if let Err(why) = video_proc_amp.GetRange(
                        id,
                        &mut min,
                        &mut max,
                        &mut step,
                        &mut default,
                        &mut flag,
                    ) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Range", control_id, control),
                            error: why.to_string(),
                        });
                    }
                    if let Err(why) = video_proc_amp.Get(id, &mut value, &mut flag) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Value", control_id, control),
                            error: why.to_string(),
                        });
                    }
                    ControlValueDescription::IntegerRange {
                        min: i64::from(min),
                        max: i64::from(max),
                        value: i64::from(value),
                        step: i64::from(step),
                        default: i64::from(default),
                    }
                },
                MFControlId::CCValue(id) => unsafe {
                    if let Err(why) = camera_control.GetRange(
                        id,
                        &mut min,
                        &mut max,
                        &mut step,
                        &mut default,
                        &mut flag,
                    ) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Range", control_id, control),
                            error: why.to_string(),
                        });
                    }
                    if let Err(why) = camera_control.Get(id, &mut value, &mut flag) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Value", control_id, control),
                            error: why.to_string(),
                        });
                    }

                    ControlValueDescription::Integer {
                        value: i64::from(value),
                        default: i64::from(default),
                        step: i64::from(step),
                    }
                },
                MFControlId::CCRange(id) => unsafe {
                    if let Err(why) = camera_control.GetRange(
                        id,
                        &mut min,
                        &mut max,
                        &mut step,
                        &mut default,
                        &mut flag,
                    ) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Range", control_id, control),
                            error: why.to_string(),
                        });
                    }
                    if let Err(why) = camera_control.Get(id, &mut value, &mut flag) {
                        return Err(NokhwaError::GetPropertyError {
                            property: format!("{:?}: {} - Value", control_id, control),
                            error: why.to_string(),
                        });
                    }
                    ControlValueDescription::IntegerRange {
                        min: i64::from(min),
                        max: i64::from(max),
                        value: i64::from(value),
                        step: i64::from(step),
                        default: i64::from(default),
                    }
                },
            };

            let is_manual = if flag == CameraControl_Flags_Manual.0 {
                KnownCameraControlFlag::Manual
            } else {
                KnownCameraControlFlag::Automatic
            };

            Ok(CameraControl::new(
                control,
                control.to_string(),
                ctrl_value_set,
                vec![is_manual],
                true,
            ))
        }

        pub fn set_control(
            &mut self,
            control: KnownCameraControl,
            value: ControlValueSetter,
        ) -> Result<(), NokhwaError> {
            let current_value = self.control(control)?;

            let camera_control = unsafe {
                let mut receiver: MaybeUninit<IAMCameraControl> = MaybeUninit::uninit();
                let ptr_receiver = receiver.as_mut_ptr();
                if let Err(why) = self.source_reader.GetServiceForStream(
                    MF_SOURCE_READER_MEDIASOURCE,
                    &GUID_NULL,
                    &IAMCameraControl::IID,
                    ptr_receiver
                        .cast::<IAMCameraControl>()
                        .cast::<*mut c_void>(),
                ) {
                    return Err(NokhwaError::SetPropertyError {
                        property: "MF_SOURCE_READER_MEDIASOURCE".to_string(),
                        value: "IAMCameraControl".to_string(),
                        error: why.to_string(),
                    });
                }
                receiver.assume_init()
            };
            let video_proc_amp = unsafe {
                let mut receiver: MaybeUninit<IAMVideoProcAmp> = MaybeUninit::uninit();
                let ptr_receiver = receiver.as_mut_ptr();
                if let Err(why) = self.source_reader.GetServiceForStream(
                    MF_SOURCE_READER_MEDIASOURCE,
                    &GUID_NULL,
                    &IAMVideoProcAmp::IID,
                    ptr_receiver.cast::<IAMVideoProcAmp>().cast::<*mut c_void>(),
                ) {
                    return Err(NokhwaError::SetPropertyError {
                        property: "MF_SOURCE_READER_MEDIASOURCE".to_string(),
                        value: "IAMVideoProcAmp".to_string(),
                        error: why.to_string(),
                    });
                }
                receiver.assume_init()
            };

            let control_id = kcc_to_i32(control).ok_or(NokhwaError::SetPropertyError {
                property: "CameraControl".to_string(),
                value: control.to_string(),
                error: "Does not exist".to_string(),
            })?;

            let ctrl_value = match value {
                ControlValueSetter::Integer(i) => i as i32,
                ControlValueSetter::Boolean(b) => i32::from(b),
                v => {
                    return Err(NokhwaError::StructureError {
                        structure: format!("ControlValueSetter {}", v),
                        error: "invalid value type".to_string(),
                    })
                }
            };

            let flag = current_value
                .flag()
                .get(0)
                .map(|x| {
                    if *x == KnownCameraControlFlag::Automatic {
                        CameraControl_Flags_Auto
                    } else {
                        CameraControl_Flags_Manual
                    }
                })
                .ok_or(NokhwaError::StructureError {
                    structure: "KnownCameraControlFlag".to_string(),
                    error: "could not cast to i32".to_string(),
                })?;

            match control_id {
                MFControlId::ProcAmpBoolean(id) | MFControlId::ProcAmpRange(id) => unsafe {
                    if let Err(why) = video_proc_amp.Set(id, ctrl_value, flag.0) {
                        return Err(NokhwaError::SetPropertyError {
                            property: control.to_string(),
                            value: ctrl_value.to_string(),
                            error: why.to_string(),
                        });
                    }
                },
                MFControlId::CCValue(id) | MFControlId::CCRange(id) => unsafe {
                    if let Err(why) = camera_control.Set(id, ctrl_value, flag.0) {
                        return Err(NokhwaError::SetPropertyError {
                            property: control.to_string(),
                            value: ctrl_value.to_string(),
                            error: why.to_string(),
                        });
                    }
                },
            }

            Ok(())
        }

        #[allow(clippy::cast_sign_loss)]
        pub fn format_refreshed(&mut self) -> Result<CameraFormat, NokhwaError> {
            match unsafe {
                self.source_reader
                    .GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32)
            } {
                Ok(media_type) => {
                    let resolution = match unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) } {
                        Ok(res) => {
                            let width = (res >> 32) as u32;
                            let height = ((res << 32) >> 32) as u32;

                            Resolution {
                                width_x: width,
                                height_y: height,
                            }
                        }
                        Err(why) => {
                            return Err(NokhwaError::GetPropertyError {
                                property: "MF_MT_FRAME_SIZE".to_string(),
                                error: why.to_string(),
                            })
                        }
                    };

                    let frame_rate = match unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE) } {
                        Ok(fps) => fps as u32,
                        Err(why) => {
                            return Err(NokhwaError::GetPropertyError {
                                property: "MF_MT_FRAME_RATE".to_string(),
                                error: why.to_string(),
                            })
                        }
                    };

                    let format = match unsafe { media_type.GetGUID(&MF_MT_SUBTYPE) } {
                        Ok(fcc) => match guid_to_frameformat(fcc) {
                            Some(ff) => ff,
                            None => {
                                return Err(NokhwaError::GetPropertyError {
                                    property: "MF_MT_SUBTYPE".to_string(),
                                    error: "Unknown".to_string(),
                                })
                            }
                        },
                        Err(why) => {
                            return Err(NokhwaError::GetPropertyError {
                                property: "MF_MT_SUBTYPE".to_string(),
                                error: why.to_string(),
                            })
                        }
                    };

                    let cfmt = CameraFormat::new(resolution, format, frame_rate);
                    self.device_format = cfmt;

                    Ok(cfmt)
                }
                Err(why) => Err(NokhwaError::GetPropertyError {
                    property: "MF_SOURCE_READER_FIRST_VIDEO_STREAM".to_string(),
                    error: why.to_string(),
                }),
            }
        }

        pub fn format(&self) -> CameraFormat {
            self.device_format
        }

        pub fn set_format(&mut self, format: CameraFormat) -> Result<(), NokhwaError> {
            // We need to make sure to use all the original attributes of the IMFMediaType to avoid problems.
            // Otherwise, constructing IMFMediaType from scratch can sometimes fail due to not exactly matching.
            // Therefore, we search for the first media_type that matches and also works correctly.

            let mut last_error: Option<NokhwaError> = None;

            let mut index = 0;
            while let Ok(media_type) = unsafe {
                self.source_reader
                    .GetNativeMediaType(MEDIA_FOUNDATION_FIRST_VIDEO_STREAM, index)
            } {
                index += 1;
                let fourcc = match unsafe { media_type.GetGUID(&MF_MT_SUBTYPE) } {
                    Ok(fcc) => fcc,
                    Err(why) => {
                        return Err(NokhwaError::GetPropertyError {
                            property: "MF_MT_SUBTYPE".to_string(),
                            error: why.to_string(),
                        })
                    }
                };

                let frame_fmt = match guid_to_frameformat(fourcc) {
                    Some(fcc) => fcc,
                    None => continue,
                };

                if frame_fmt != format.format() {
                    continue;
                }

                let (width, height) = match unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) } {
                    Ok(res_u64) => {
                        let width = (res_u64 >> 32) as u32;
                        let height = res_u64 as u32; // the cast will truncate the upper bits
                        (width, height)
                    }
                    Err(why) => {
                        return Err(NokhwaError::GetPropertyError {
                            property: "MF_MT_FRAME_SIZE".to_string(),
                            error: why.to_string(),
                        })
                    }
                };

                if (Resolution {
                    width_x: width,
                    height_y: height,
                }) != format.resolution()
                {
                    continue;
                }

                // MFRatio is represented as 2 u32s in memory. This means we can convert it to 2
                let framerate_list = {
                    let mut framerates = vec![0_u32; 3];
                    if let Ok(fraction_u64) =
                        unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE_RANGE_MAX) }
                    {
                        let mut numerator = (fraction_u64 >> 32) as u32;
                        let denominator = fraction_u64 as u32;
                        if denominator != 1 {
                            numerator = 0;
                        }
                        framerates.push(numerator);
                    };
                    if let Ok(fraction_u64) = unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE) } {
                        let mut numerator = (fraction_u64 >> 32) as u32;
                        let denominator = fraction_u64 as u32;
                        if denominator != 1 {
                            numerator = 0;
                        }
                        framerates.push(numerator);
                    };
                    if let Ok(fraction_u64) =
                        unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE_RANGE_MIN) }
                    {
                        let mut numerator = (fraction_u64 >> 32) as u32;
                        let denominator = fraction_u64 as u32;
                        if denominator != 1 {
                            numerator = 0;
                        }
                        framerates.push(numerator);
                    };
                    framerates
                };

                for frame_rate in framerate_list {
                    if frame_rate == format.frame_rate() {
                        let result = unsafe {
                            self.source_reader.SetCurrentMediaType(
                                MEDIA_FOUNDATION_FIRST_VIDEO_STREAM,
                                None,
                                &media_type,
                            )
                        };

                        match result {
                            Ok(_) => {
                                self.device_format = format;
                                self.format_refreshed()?;
                                return Ok(());
                            }
                            Err(why) => {
                                last_error = Some(NokhwaError::SetPropertyError {
                                    property: "MEDIA_FOUNDATION_FIRST_VIDEO_STREAM".to_string(),
                                    value: format!("{media_type:?}"),
                                    error: why.to_string(),
                                });
                            }
                        }
                    }
                }
            }

            if let Some(err) = last_error {
                return Err(err);
            }

            Err(NokhwaError::InitializeError {
                backend: ApiBackend::MediaFoundation,
                error: "Failed to fulfill requested format".to_string(),
            })
        }

        pub fn is_stream_open(&self) -> bool {
            self.is_open.get()
        }

        pub fn start_stream(&mut self) -> Result<(), NokhwaError> {
            if let Err(why) = unsafe {
                self.source_reader
                    .SetStreamSelection(MEDIA_FOUNDATION_FIRST_VIDEO_STREAM, true)
            } {
                return Err(NokhwaError::OpenStreamError(why.to_string()));
            }

            self.stream_epoch = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok();
            self.is_open.set(true);
            Ok(())
        }

        pub fn raw_bytes(&mut self) -> Result<(Cow<'_, [u8]>, Option<Duration>), NokhwaError> {
            // ReadSample supplies an owned reference. Preallocating a sample
            // here leaks it when the COM out-parameter overwrites the pointer.
            let mut imf_sample: Option<IMFSample> = None;
            let mut stream_flags = 0;
            let mut sample_time_100ns: i64 = 0;
            {
                loop {
                    if let Err(why) = unsafe {
                        self.source_reader.ReadSample(
                            MEDIA_FOUNDATION_FIRST_VIDEO_STREAM,
                            0,
                            None,
                            Some(&mut stream_flags),
                            Some(&mut sample_time_100ns),
                            Some(&mut imf_sample),
                        )
                    } {
                        return Err(NokhwaError::ReadFrameError(why.to_string()));
                    }

                    if stream_flags
                        & (MF_SOURCE_READERF_ENDOFSTREAM.0 | MF_SOURCE_READERF_ERROR.0) as u32
                        != 0
                    {
                        return Err(NokhwaError::ReadFrameError(format!(
                            "Source reader ended or failed (flags: {stream_flags:#x})"
                        )));
                    }
                    if imf_sample.is_some() {
                        break;
                    }
                }
            }

            let imf_sample = match imf_sample {
                Some(sample) => sample,
                None => {
                    // shouldn't happen
                    return Err(NokhwaError::ReadFrameError("No sample".to_string()));
                }
            };

            // Calculate absolute capture timestamp.
            let capture_ts = if sample_time_100ns > 0 {
                let sample_offset = Duration::from_nanos(sample_time_100ns as u64 * 100);
                self.stream_epoch
                    .and_then(|epoch| epoch.checked_add(sample_offset))
            } else {
                None
            };

            let buffer = match unsafe { imf_sample.ConvertToContiguousBuffer() } {
                Ok(buf) => buf,
                Err(why) => return Err(NokhwaError::ReadFrameError(why.to_string())),
            };

            let mut buffer_valid_length = 0;
            let mut buffer_start_ptr = std::ptr::null_mut::<u8>();

            if let Err(why) =
                unsafe { buffer.Lock(&mut buffer_start_ptr, None, Some(&mut buffer_valid_length)) }
            {
                return Err(NokhwaError::ReadFrameError(why.to_string()));
            }

            let result = if buffer_start_ptr.is_null() || buffer_valid_length == 0 {
                Err(NokhwaError::ReadFrameError(
                    "Empty sample buffer".to_string(),
                ))
            } else {
                // The bytes remain valid until Unlock; copy while locked.
                Ok(unsafe {
                    from_raw_parts(buffer_start_ptr, buffer_valid_length as usize).to_vec()
                })
            };
            // Balance Lock on both valid and empty samples. A locked capture
            // buffer cannot safely be recycled by the driver's buffer pool.
            unsafe { buffer.Unlock() }
                .map_err(|error| NokhwaError::ReadFrameError(error.to_string()))?;
            let data_slice = result?;

            Ok((Cow::from(data_slice), capture_ts))
        }

        pub fn stop_stream(&mut self) {
            self.stream_epoch = None;
            self.is_open.set(false);
        }
    }

    impl Drop for MediaFoundationDevice {
        fn drop(&mut self) {
            unsafe {
                let _ = self
                    .source_reader
                    .Flush(MEDIA_FOUNDATION_FIRST_VIDEO_STREAM);
            }
            // Fields drop in declaration order: reader, source (Shutdown),
            // then runtime (MFShutdown and this thread's CoUninitialize).
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use windows::Win32::System::Com::{
            CoGetApartmentType, APTTYPE, APTTYPEQUALIFIER, APTTYPE_MAINSTA, APTTYPE_STA,
            COINIT_MULTITHREADED,
        };

        fn apartment() -> Option<(APTTYPE, APTTYPEQUALIFIER)> {
            let mut apartment = APTTYPE::default();
            let mut qualifier = APTTYPEQUALIFIER::default();
            unsafe { CoGetApartmentType(&mut apartment, &mut qualifier) }.ok()?;
            Some((apartment, qualifier))
        }

        fn sta_initialized() -> bool {
            apartment().is_some_and(|(kind, _)| kind == APTTYPE_STA || kind == APTTYPE_MAINSTA)
        }

        fn create_attributes() {
            let mut attributes = None;
            unsafe { MFCreateAttributes(&mut attributes, 1) }.unwrap();
            assert!(attributes.is_some());
        }

        #[test]
        fn runtime_initializes_each_thread_and_balances_nested_owners() {
            std::thread::spawn(|| {
                assert!(!sta_initialized());
                let first = MfRuntime::new().unwrap();
                let second = MfRuntime::new().unwrap();
                drop(first);
                assert!(sta_initialized());
                create_attributes();
                let (ready_tx, ready_rx) = std::sync::mpsc::channel();
                let (resume_tx, resume_rx) = std::sync::mpsc::channel();
                let worker = std::thread::spawn(move || {
                    assert!(!sta_initialized());
                    let runtime = MfRuntime::new().unwrap();
                    assert!(sta_initialized());
                    ready_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    create_attributes();
                    drop(runtime);
                    assert!(!sta_initialized());
                });
                ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                drop(second);
                assert!(!sta_initialized());
                resume_tx.send(()).unwrap();
                worker.join().unwrap();
            })
            .join()
            .unwrap();
        }

        #[test]
        fn runtime_preserves_an_existing_mta_apartment() {
            std::thread::spawn(|| {
                unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
                    .ok()
                    .unwrap();
                let before = apartment();
                let runtime = MfRuntime::new().unwrap();
                create_attributes();
                drop(runtime);
                assert_eq!(apartment(), before);
                unsafe { CoUninitialize() };
            })
            .join()
            .unwrap();
        }

        #[test]
        fn enumeration_releases_its_runtime_on_return() {
            std::thread::spawn(|| {
                for _ in 0..3 {
                    query_media_foundation_descriptors().unwrap();
                    assert!(!sta_initialized());
                }
            })
            .join()
            .unwrap();
        }
    }
}

#[cfg(any(not(windows), feature = "docs-only"))]
#[allow(clippy::missing_errors_doc)]
#[allow(clippy::unused_self)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::must_use_candidate)]
pub mod wmf {
    use nokhwa_core::error::NokhwaError;
    use nokhwa_core::types::{
        CameraControl, CameraFormat, CameraIndex, CameraInfo, ControlValueSetter,
        KnownCameraControl,
    };
    use std::{borrow::Cow, time::Duration};

    pub fn initialize_mf() -> Result<(), NokhwaError> {
        Err(NokhwaError::NotImplementedError(
            "Not on windows".to_string(),
        ))
    }

    pub fn de_initialize_mf() -> Result<(), NokhwaError> {
        Err(NokhwaError::NotImplementedError(
            "Not on windows".to_string(),
        ))
    }

    pub fn query_msmf() -> Result<Vec<CameraInfo>, NokhwaError> {
        Err(NokhwaError::NotImplementedError(
            "Not on windows".to_string(),
        ))
    }

    pub struct MediaFoundationDevice {
        camera: CameraIndex,
    }

    impl MediaFoundationDevice {
        pub fn new(_index: CameraIndex) -> Result<Self, NokhwaError> {
            Ok(MediaFoundationDevice {
                camera: CameraIndex::Index(0),
            })
        }

        pub fn index(&self) -> &CameraIndex {
            &self.camera
        }

        pub fn name(&self) -> String {
            String::new()
        }

        pub fn symlink(&self) -> String {
            String::new()
        }

        pub fn compatible_format_list(&mut self) -> Result<Vec<CameraFormat>, NokhwaError> {
            Err(NokhwaError::NotImplementedError(
                "Only on Windows".to_string(),
            ))
        }

        pub fn control(&self, _control: KnownCameraControl) -> Result<CameraControl, NokhwaError> {
            Err(NokhwaError::NotImplementedError(
                "Only on Windows".to_string(),
            ))
        }

        pub fn set_control(
            &mut self,
            _control: KnownCameraControl,
            _value: ControlValueSetter,
        ) -> Result<(), NokhwaError> {
            Err(NokhwaError::NotImplementedError(
                "Only on Windows".to_string(),
            ))
        }

        pub fn format_refreshed(&mut self) -> Result<CameraFormat, NokhwaError> {
            Err(NokhwaError::NotImplementedError(
                "Only on Windows".to_string(),
            ))
        }

        pub fn format(&self) -> CameraFormat {
            CameraFormat::default()
        }

        pub fn set_format(&mut self, _format: CameraFormat) -> Result<(), NokhwaError> {
            Err(NokhwaError::NotImplementedError(
                "Only on Windows".to_string(),
            ))
        }

        pub fn is_stream_open(&self) -> bool {
            false
        }

        pub fn start_stream(&mut self) -> Result<(), NokhwaError> {
            Err(NokhwaError::NotImplementedError(
                "Only on Windows".to_string(),
            ))
        }

        pub fn raw_bytes(&mut self) -> Result<(Cow<'_, [u8]>, Option<Duration>), NokhwaError> {
            Err(NokhwaError::NotImplementedError(
                "Only on Windows".to_string(),
            ))
        }

        pub fn stop_stream(&mut self) {}
    }

    impl Drop for MediaFoundationDevice {
        fn drop(&mut self) {}
    }
}
