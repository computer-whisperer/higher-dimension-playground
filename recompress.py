
import os
import Imath
import OpenEXR
import numpy as np

def convert_exr_compression(input_folder, output_folder):
    """
    Convert PIZ-compressed EXR images to RLE-compressed EXR images.

    Parameters:
    - input_folder (str): Path to the folder containing input PIZ-compressed EXR files
    - output_folder (str): Path to the folder where RLE-compressed EXR files will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.exr'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Open the input EXR file
                exr_file = OpenEXR.InputFile(input_path)

                # Get image header and data window
                header = exr_file.header()
                dw = header['dataWindow']

                # Read the pixels
                channels = header['channels']
                pixel_types = {}
                arrays = {}

                # Read each channel
                for name, channel in channels.items():
                    pixel_type = channel.type
                    pixel_types[name] = pixel_type
                    arrays[name] = np.frombuffer(
                        exr_file.channel(name, pixel_type),
                        dtype=np.float32
                    ).reshape((dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1))

                # Close input file
                exr_file.close()

                # Create output header with RLE compression
                out_header = header.copy()
                out_header['compression'] = Imath.Compression(Imath.Compression.RLE_COMPRESSION)

                # Create output EXR file with RLE compression
                output_file = OpenEXR.OutputFile(output_path, out_header)

                # Write channels
                for name, array in arrays.items():
                    output_file.writePixels({name: array.astype(np.float32)})

                # Close output file
                output_file.close()

                print(f"Converted {filename} to RLE compression")

            except Exception as e:
                print(f"Error converting {filename}: {e}")

def main():
    # Example usage
    input_folder = "/home/christian/Documents/tesseract_images"
    output_folder = "/home/christian/Documents/tesseract_images_converted"

    convert_exr_compression(input_folder, output_folder)

if __name__ == "__main__":
    main()