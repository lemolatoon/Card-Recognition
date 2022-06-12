use opencv::imgcodecs;
use opencv::prelude::MatTraitConst;
use std::env;
use std::fs::ReadDir;
use std::io::Error;
use std::os;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);

    if let Some(arg1) = args.get(1) {
        if arg1 == "back" {
            return make_datasets().unwrap();
        }
    } else {
        println!("Do nothing...");
    }
}

fn make_datasets() -> Result<(), opencv::Error> {
    println!("Making Datasets....");
    const BACKGROUND_IMG_PATH: &str = "../images/background/root";
    const CARDS_IMG_PATH: &str = "../pdf2img/img";
    const DATASETS_OUT_PATH: &str = "../images/datasets/root";
    let abs_back_img_path = get_absolute_path(Path::new(BACKGROUND_IMG_PATH));
    let abs_cards_img_path = get_absolute_path(Path::new(CARDS_IMG_PATH));
    let abs_data_out_path = get_absolute_path(Path::new(DATASETS_OUT_PATH));

    let back_file_iter = FilePathIterator::new(abs_back_img_path).unwrap();

    for back_filename in back_file_iter {
        let img = imgcodecs::imread(&back_filename, imgcodecs::IMREAD_COLOR)?;
        println!("{:?}", img.cols());
        println!("{:?}", img.col(0).map(|mat| mat.row(0)));
        println!("{:?}", img.rows());
        println!("{:?}", img.row(0));
        println!("{:?}", img);
        let vec = opencv::core::Vector::new();
        imgcodecs::imwrite("fig.jpg", &img, &vec)?;
    }

    Ok(())
}

struct FilePathIterator {
    pathbuf_iter: ReadDir,
}

impl FilePathIterator {
    fn new(root: PathBuf) -> Option<Self> {
        if let Ok(read_dir) = root.read_dir() {
            return Some(Self {
                pathbuf_iter: read_dir,
            });
        } else {
            return None;
        }
    }
}

impl Iterator for FilePathIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(entry) = self.pathbuf_iter.next()? {
            let pathbuf = entry.path();
            let filename = pathbuf.as_path().to_str()?;
            return Some(filename.to_string());
        }
        return None;
    }
}

fn get_absolute_path(path: &Path) -> PathBuf {
    let pwd = env::current_dir().unwrap();
    return pwd.join(path);
}
