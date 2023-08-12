import cv2
import glob

from Pix2Pix.pix2pix_model_loader import create_pix2pix_model 
from Pix2Pix.opt import Pix2Pix_OPT 
from Pix2Pix.test import *
from utils.utility import *

pix2pix_model_path = './weights'

# Pix2Pix Module
pix2pixModel = create_pix2pix_model(Pix2Pix_OPT(pix2pix_model_path))

def process_image(img_path):
    try:
        # Load the image using OpenCV
        frame= cv2.imread(img_path)

        print(frame.shape, img_path)

        pix2pixOutput = pix2pixEval(pix2pixModel, frame)
        camera_test = pix2pixOutput['fake_B']
        camera_test = cv2.cvtColor(camera_test,cv2.COLOR_RGB2BGR)

        concatenated_img = cv2.hconcat([cv2.resize(frame, (1024,1024)), camera_test])

        cv2.imshow("result  ", concatenated_img)
        cv2.waitKey(1)
        return concatenated_img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def main(input_folder, output_folder):
    # Traverse through the input folder
    for img_path in glob.glob(input_folder + '/**/*.png', recursive=True):
        # Process the image
        processed_img = process_image(img_path)
        
        if processed_img is not None:
            # Create the corresponding output directory if it doesn't exist
            relative_path = os.path.relpath(os.path.dirname(img_path), input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the processed image in the output directory
            output_img_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_img_path, processed_img)

if __name__ == "__main__":
    input_folder = "data"
    output_folder = "out"
    main(input_folder, output_folder)
