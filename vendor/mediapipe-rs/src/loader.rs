//! Finding, fetching and opening `libmediapipe`.
//!
//! The library is ~34 MB, so it cannot be vendored into the crate, and it is
//! opened with `dlopen` rather than linked so that this crate builds anywhere —
//! including docs.rs and machines that have never seen MediaPipe.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::error::{Error, Result};
use crate::sys::{self, Abi};

/// The MediaPipe release this crate's bindings and download table are pinned to.
pub const MEDIAPIPE_VERSION: &str = "0.10.35";

/// Where the loaded library came from. Reported in diagnostics so a stale cache
/// is distinguishable from an explicit override.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibrarySource {
    /// `$MEDIAPIPE_LIB`
    Env(PathBuf),
    /// Previously downloaded into the user cache.
    Cache(PathBuf),
    /// Just fetched from the official PyPI wheel.
    Downloaded(PathBuf),
}

impl LibrarySource {
    pub fn path(&self) -> &Path {
        match self {
            LibrarySource::Env(p) | LibrarySource::Cache(p) | LibrarySource::Downloaded(p) => p,
        }
    }
}

pub struct Lib {
    pub raw: sys::MpLib,
    pub abi: Abi,
    pub source: LibrarySource,
}

impl std::fmt::Debug for Lib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `raw` is a table of 119 function pointers; nobody wants to read it.
        f.debug_struct("Lib")
            .field("abi", &self.abi)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

static LIB: OnceLock<Lib> = OnceLock::new();
static LOAD_LOCK: Mutex<()> = Mutex::new(());

/// Loads `libmediapipe` on first use and returns the shared handle.
///
/// Errors are not cached: a failed load can be retried after, say, the user sets
/// `$MEDIAPIPE_LIB`.
pub fn lib() -> Result<&'static Lib> {
    if let Some(l) = LIB.get() {
        return Ok(l);
    }
    // Serialised so a race cannot start two 12 MB downloads.
    let _guard = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(l) = LIB.get() {
        return Ok(l);
    }
    let loaded = load()?;
    Ok(LIB.get_or_init(|| loaded))
}

fn load() -> Result<Lib> {
    let mut searched = Vec::new();

    let source = if let Some(p) = std::env::var_os("MEDIAPIPE_LIB") {
        let p = PathBuf::from(p);
        if !p.exists() {
            searched.push(p.clone());
            return Err(Error::LibraryNotFound { searched });
        }
        LibrarySource::Env(p)
    } else {
        let cached = cache_path()?;
        if cached.exists() {
            LibrarySource::Cache(cached)
        } else {
            searched.push(cached.clone());
            LibrarySource::Downloaded(fetch(&cached)?)
        }
    };

    // SAFETY: dlopen runs the library's initialisers. libmediapipe is a normal
    // shared object with no unusual init behaviour.
    let raw = unsafe { sys::MpLib::new(source.path()) }.map_err(|e| Error::Load {
        path: source.path().to_path_buf(),
        source: e,
    })?;

    // See `sys::Abi`: post-rename builds renamed InteractiveSegmenter to
    // ...Legacy, so the symbol's presence dates the library across the one
    // struct-layout change that matters. It is a heuristic on an unrelated
    // symbol, so $MEDIAPIPE_ABI can override it if upstream ever reshuffles
    // those names again.
    let abi = match std::env::var("MEDIAPIPE_ABI").as_deref() {
        Ok("v0_10_35") => Abi::V0_10_35,
        Ok("renamed") => Abi::Renamed,
        Ok(other) => {
            return Err(Error::Download(format!(
                "MEDIAPIPE_ABI must be `v0_10_35` or `renamed`, got `{other}`"
            )));
        }
        Err(_) if raw.MpInteractiveSegmenterLegacyCreate.is_ok() => Abi::Renamed,
        Err(_) => Abi::V0_10_35,
    };

    Ok(Lib { raw, abi, source })
}

fn cache_dir() -> Result<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    };
    Ok(base
        .unwrap_or_else(std::env::temp_dir)
        .join("mediapipe-rs")
        .join(MEDIAPIPE_VERSION))
}

fn cache_path() -> Result<PathBuf> {
    Ok(cache_dir()?.join(TARGET.lib_file_name))
}

/// A prebuilt library published for one target, with the hash of the wheel that
/// carries it. Hashes are baked in at compile time rather than fetched, so a
/// compromised index cannot swap the binary.
struct Target {
    lib_file_name: &'static str,
    #[cfg_attr(not(feature = "download"), allow(dead_code))]
    wheel_url: &'static str,
    #[cfg_attr(not(feature = "download"), allow(dead_code))]
    wheel_sha256: [u8; 32],
}

const fn hex(s: &[u8; 64]) -> [u8; 32] {
    const fn nib(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("sha256 literal must be lowercase hex"),
        }
    }
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (nib(s[i * 2]) << 4) | nib(s[i * 2 + 1]);
        i += 1;
    }
    out
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const TARGET: Target = Target {
    lib_file_name: "libmediapipe.so",
    wheel_url: "https://files.pythonhosted.org/packages/32/8f/1bc57dbc9b7b03c8f875aac23380ec57e9002cc02fe6720045fb263f3966/mediapipe-0.10.35-py3-none-manylinux_2_28_x86_64.whl",
    wheel_sha256: hex(b"db9a579df48cffe9570cd3e93f6a5d2dd089a1103b846c60c5b5de8a21c38db0"),
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const TARGET: Target = Target {
    lib_file_name: "libmediapipe.dylib",
    wheel_url: "https://files.pythonhosted.org/packages/aa/c2/439e948d2a9a542498aee5d8fa3fb91ac6f4478be5d502f0c78ca3fb2333/mediapipe-0.10.35-py3-none-macosx_11_0_arm64.whl",
    wheel_sha256: hex(b"3b31376f34ca3665e34b834565996464cd66c9c91316e914fa7f149c891ce7ac"),
};

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const TARGET: Target = Target {
    lib_file_name: "libmediapipe.dll",
    wheel_url: "https://files.pythonhosted.org/packages/5b/f6/763477e9aeed98accc984ed6ee3f11a21a0c5fd1d1c6586b8d07067748ff/mediapipe-0.10.35-py3-none-win_amd64.whl",
    wheel_sha256: hex(b"b08f001cf3c3cd0d88d9ed68f3368dc8a4913f568281a93117f083115aa672ba"),
};

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
const TARGET: Target = Target {
    lib_file_name: "libmediapipe.dll",
    wheel_url: "https://files.pythonhosted.org/packages/07/b3/5c7fa594c731e8dafab9f1a46ab6cef670fa62dbbfb6248cc70e42ec6fc5/mediapipe-0.10.35-py3-none-win_arm64.whl",
    wheel_sha256: hex(b"46255326a6213118aaa518a7aa25e35f93337e82677960cc2a945f117bff8444"),
};

// Notably absent: linux-aarch64. Google publishes no manylinux aarch64 wheel, so
// Raspberry Pi / Jetson users must build libmediapipe from source.
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64"),
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "aarch64"),
)))]
const TARGET: Target = Target {
    lib_file_name: "libmediapipe.so",
    wheel_url: "",
    wheel_sha256: [0; 32],
};

const TARGET_SUPPORTED: bool = cfg!(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64"),
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "aarch64"),
));

#[cfg(not(feature = "download"))]
fn fetch(target_path: &Path) -> Result<PathBuf> {
    if !TARGET_SUPPORTED {
        return Err(Error::UnsupportedTarget {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        });
    }
    Err(Error::LibraryNotFound {
        searched: vec![target_path.to_path_buf()],
    })
}

/// Downloads the official wheel, verifies its hash, and extracts just the
/// library into `dest`.
#[cfg(feature = "download")]
fn fetch(dest: &Path) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};

    if !TARGET_SUPPORTED {
        return Err(Error::UnsupportedTarget {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        });
    }

    let wheel = ureq::get(TARGET.wheel_url)
        .call()
        .map_err(|e| Error::Download(format!("GET {}: {e}", TARGET.wheel_url)))?
        .body_mut()
        .with_config()
        .limit(64 * 1024 * 1024)
        .read_to_vec()
        .map_err(|e| Error::Download(format!("reading {}: {e}", TARGET.wheel_url)))?;

    let digest: [u8; 32] = Sha256::digest(&wheel).into();
    if digest != TARGET.wheel_sha256 {
        return Err(Error::ChecksumMismatch {
            expected: to_hex(&TARGET.wheel_sha256),
            got: to_hex(&digest),
        });
    }

    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(wheel))
        .map_err(|e| Error::Download(format!("wheel is not a valid zip: {e}")))?;
    let index = (0..zip.len())
        .find(|&i| {
            zip.by_index(i)
                .ok()
                .is_some_and(|f| f.name().ends_with(TARGET.lib_file_name))
        })
        .ok_or_else(|| Error::Download(format!("wheel contains no {}", TARGET.lib_file_name)))?;

    let dir = dest
        .parent()
        .expect("cache path should have a parent, it is built by joining onto a cache dir");
    std::fs::create_dir_all(dir)?;

    // Write beside the destination and rename, so a torn download or a
    // concurrent process never leaves a half-written library in place.
    let tmp = dest.with_extension(format!("tmp{}", std::process::id()));
    let mut out = std::fs::File::create(&tmp)?;
    std::io::copy(
        &mut zip
            .by_index(index)
            .map_err(|e| Error::Download(format!("extracting: {e}")))?,
        &mut out,
    )?;
    drop(out);
    std::fs::rename(&tmp, dest)?;

    Ok(dest.to_path_buf())
}

#[cfg(feature = "download")]
fn to_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
