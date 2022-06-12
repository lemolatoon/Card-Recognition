use std::env;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    let path = Path::new("build");
    println!("{:?}", get_absolute_path(path));
    println!("{:?}", env::var("PKG_CONFIG_PATH"));
    println!("{:?}", env::var("LD_LIBRARY_PATH"));
}

fn get_absolute_path(path: &Path) -> PathBuf {
    let pwd = env::current_dir().unwrap();
    return pwd.join(path);
}
