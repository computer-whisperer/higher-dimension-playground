use std::fs;
use exr::prelude::WritableImage;

fn main() {
    let input_dir = "/home/christian/Documents/tesseract_images/";
    let output_dir = "/home/christian/Documents/tesseract_images_converted/";

    let files = fs::read_dir(input_dir).unwrap();

    for file_name in files {
        if let Ok(file_name) = file_name {
            let mut dest_path = output_dir.to_string();
            dest_path.push_str(file_name.file_name().to_str().unwrap());

            if !fs::metadata(&dest_path).is_ok() {
                let mut image = exr::prelude::read_all_data_from_file(file_name.path());

                match image {
                    Ok(mut image) => {
                        for layer in image.layer_data.as_mut() {
                            layer.encoding = exr::prelude::Encoding::FAST_LOSSLESS;
                        }

                        image.write().to_file(dest_path).unwrap();
                        println!("Converted: {}", file_name.file_name().to_str().unwrap());
                    },
                    Err(error) => {
                        eprintln!("Error reading EXR file: {}", error);
                        continue;
                    }
                }


            }


        }
    }

    ()
}