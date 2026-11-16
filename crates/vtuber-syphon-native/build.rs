fn main() {
    println!("cargo:rerun-if-changed=src/sender.m");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    cc::Build::new()
        .file("src/sender.m")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("rustuberv_syphon");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}
