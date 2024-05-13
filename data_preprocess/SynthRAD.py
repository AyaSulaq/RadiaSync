###Best Pre-Processing till now###
#To make sure that the source folders are ct and mri and the target folders CT and MRI
import numpy as np
import os
import argparse

from scipy import ndimage



import tqdm
import glob
import re

from common import read_img_sitk

# Normalization Function has been added -> Accuracy reached 80s

def normalize_image(image):
    """Normalize the pixel values of the image to the range [0, 1]."""
    image_min = image.min()
    image_max = image.max()
    normalized_image = (image - image_min) / (image_max - image_min)
    return normalized_image



def read_multimodal(src_path, dst_path, series_map):
    pattern = r"\.nii.gz$"
    for src_mode, dst_mode in series_map.items():
        print('process: ' + src_mode)
        files = glob.glob("%s/%s/*" % (src_path, src_mode))

        for f in tqdm.tqdm(files):
             data = read_img_sitk(f)
             if data.shape[0] > 100:
                 normal_data = normalize_image(data)
                 name = re.sub(pattern, ".npy", f.split('/')[-1])
                 np.save(dst_path + '/' + dst_mode + '/' + name, normal_data)

def dataset_preprocess(src_path, dst_path):
    series_map = {'CT': 'CT', 'MRI': 'MRI'}
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for _, dst_mode in series_map.items():
        if not os.path.exists(os.path.join(dst_path, dst_mode)):
            os.mkdir(os.path.join(dst_path, dst_mode))

    read_multimodal(src_path, dst_path, series_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch SynthRAD")
    parser.add_argument("--data_path", default="/home/user/FedMed-GAN-main/Dataset", nargs='+', type=str, help="path to train data")
    parser.add_argument("--generated_path", default="/home/user/FedMed-GAN-main/synthrad", nargs='+', type=str, help="path to target train data")
    args = parser.parse_args()

    dataset_preprocess(src_path=args.data_path, dst_path=args.generated_path)


