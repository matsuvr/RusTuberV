//! Licenses of everything this application is built from and ships.
//!
//! `build.rs` collects the crates listed in `Cargo.lock` into
//! [`DEPENDENCY_LICENSE_GROUPS`](crate::licenses::DEPENDENCY_LICENSE_GROUPS).
//! The assets this repository ships next to the executable are listed here,
//! because they are not in the dependency graph.
//! Every license body is the verbatim upstream file. The UI only localizes the
//! headings around it; a license body is never translated.

/// Inventory generated from `Cargo.lock` and the crate sources on this machine.
mod generated {
    include!(concat!(env!("OUT_DIR"), "/oss_licenses.rs"));
}

pub use generated::DEPENDENCY_LICENSE_GROUPS;
pub use generated::{LicenseGroup, LicenseText, LicensedItem};

/// Items this repository ships itself, which `Cargo.lock` does not describe.
static BUNDLED_LICENSE_GROUPS: &[LicenseGroup] = &[
    LicenseGroup {
        expression: "Apache-2.0",
        items: &[LicensedItem {
            name: "MyoArm anatomical parameters (adapted)",
            version: "93b0ca8",
        }],
        texts: &[
            LicenseText {
                file_name: "vendor/mediapipe-rs/LICENSE",
                body: include_str!("../../../vendor/mediapipe-rs/LICENSE"),
            },
            LicenseText {
                file_name: "THIRD_PARTY_NOTICES.md",
                body: include_str!("../../../THIRD_PARTY_NOTICES.md"),
            },
        ],
    },
    LicenseGroup {
        expression: "MIT",
        items: &[LicensedItem {
            name: "Daniel Holden Joint Limits method (adapted)",
            version: "",
        }],
        texts: &[
            LicenseText {
                file_name: "LICENSE",
                body: include_str!("../../../LICENSE"),
            },
            LicenseText {
                file_name: "THIRD_PARTY_NOTICES.md",
                body: include_str!("../../../THIRD_PARTY_NOTICES.md"),
            },
        ],
    },
    LicenseGroup {
        expression: "MIT",
        items: &[LicensedItem {
            name: "RusTuberV",
            version: env!("CARGO_PKG_VERSION"),
        }],
        texts: &[LicenseText {
            file_name: "LICENSE",
            body: include_str!("../../../LICENSE"),
        }],
    },
    LicenseGroup {
        expression: "OFL-1.1",
        items: &[
            LicensedItem {
                name: "LINESeedJP_A_OTF_Rg.otf",
                version: "",
            },
            LicensedItem {
                name: "LINESeedKR-Rg.otf",
                version: "",
            },
            LicensedItem {
                name: "NotoSansCJKsc-VF.otf",
                version: "",
            },
        ],
        texts: &[
            LicenseText {
                file_name: "LICENSE-LINESeed.txt",
                body: include_str!("../../../assets/fonts/LICENSE-LINESeed.txt"),
            },
            LicenseText {
                file_name: "LICENSE-NotoSansCJK.txt",
                // Copyright notice from the bundled font's name table;
                // the upstream license file contains only the OFL text.
                body: concat!(
                    "© 2014-2021 Adobe (http://www.adobe.com/), with Reserved Font Name 'Source'.\n\n",
                    include_str!("../../../assets/fonts/LICENSE-NotoSansCJK.txt"),
                ),
            },
        ],
    },
    LicenseGroup {
        expression: "Apache-2.0",
        items: &[LicensedItem {
            name: "mediapipe (vendor/mediapipe-rs)",
            version: "",
        }],
        texts: &[
            LicenseText {
                file_name: "vendor/mediapipe-rs/LICENSE",
                body: include_str!("../../../vendor/mediapipe-rs/LICENSE"),
            },
            LicenseText {
                file_name: "vendor/mediapipe-rs/NOTICE",
                body: include_str!("../../../vendor/mediapipe-rs/NOTICE"),
            },
        ],
    },
    LicenseGroup {
        expression: "Apache-2.0",
        items: &[LicensedItem {
            name: "assets/models (MediaPipe models)",
            version: "",
        }],
        texts: &[LicenseText {
            file_name: "LICENSE.mediapipe.txt",
            body: include_str!("../../../assets/models/LICENSE.mediapipe.txt"),
        }],
    },
];

/// Assets shipped by this repository, before the dependency groups.
#[must_use]
pub fn bundled_groups() -> &'static [LicenseGroup] {
    BUNDLED_LICENSE_GROUPS
}

/// Crates from outside this repository, built from `Cargo.lock`.
#[must_use]
pub fn dependency_groups() -> &'static [LicenseGroup] {
    DEPENDENCY_LICENSE_GROUPS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_groups() -> impl Iterator<Item = &'static LicenseGroup> {
        bundled_groups().iter().chain(dependency_groups().iter())
    }

    #[test]
    fn every_group_lists_the_items_it_covers() {
        for group in all_groups() {
            assert!(!group.expression.is_empty(), "a group needs an expression");
            assert!(!group.items.is_empty(), "a group needs at least one item");
            for item in group.items {
                assert!(!item.name.is_empty());
            }
        }
    }

    #[test]
    fn every_bundled_asset_ships_its_own_license_text() {
        for group in bundled_groups() {
            assert!(
                !group.texts.is_empty(),
                "{} ships without a license text",
                group.expression
            );
            for text in group.texts {
                assert!(!text.file_name.is_empty());
                assert!(!text.body.is_empty());
            }
        }
    }

    #[test]
    fn the_shipped_dependency_set_is_listed() {
        let dependencies: Vec<&LicenseGroup> = dependency_groups().iter().collect();
        let items: usize = dependencies.iter().map(|group| group.items.len()).sum();
        assert!(
            items > 100,
            "the workspace builds from far more than {items} third-party crates"
        );
        let names: Vec<&str> = dependencies
            .iter()
            .flat_map(|group| group.items.iter().map(|item| item.name))
            .collect();
        assert!(names.contains(&"bevy"));
        assert!(names.contains(&"bevy_vrm1"));
        assert!(
            !names.contains(&"vtuber-app"),
            "this repository is not third-party"
        );
    }

    #[test]
    fn the_bundled_fonts_and_models_are_listed() {
        let names: Vec<&str> = bundled_groups()
            .iter()
            .flat_map(|group| group.items.iter().map(|item| item.name))
            .collect();
        assert!(names.contains(&"RusTuberV"));
        assert!(names.contains(&"LINESeedJP_A_OTF_Rg.otf"));
        assert!(names.contains(&"LINESeedKR-Rg.otf"));
        assert!(names.contains(&"NotoSansCJKsc-VF.otf"));
        assert!(names.contains(&"mediapipe (vendor/mediapipe-rs)"));
        assert!(names.contains(&"assets/models (MediaPipe models)"));
    }
}
