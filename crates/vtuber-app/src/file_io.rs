//! Same-directory file replacement for settings and managed model files.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Writes a unique temporary file, then replaces the destination without deleting it first.
pub(crate) fn replace_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    fs::create_dir_all(directory)?;
    let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
    temporary.write_all(contents)?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn replaces_existing_bytes_and_cleans_up_a_refused_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("saved");
        replace_file(&path, b"first").unwrap();
        replace_file(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");

        let blocked = directory.path().join("directory");
        fs::create_dir(&blocked).unwrap();
        let retained = blocked.join("retained");
        fs::write(&retained, b"existing").unwrap();
        assert!(replace_file(&blocked, b"replacement").is_err());
        assert_eq!(fs::read(&retained).unwrap(), b"existing");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
    }

    #[test]
    fn failed_write_keeps_the_existing_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let original = directory.path().join("saved");
        fs::write(&original, b"existing").unwrap();
        assert!(replace_file(&original.join("saved"), b"replacement").is_err());
        assert_eq!(fs::read(&original).unwrap(), b"existing");
    }
}
