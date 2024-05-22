import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from arch_centralized import cyclegan
from data_io.base_class import ToTensor
import argparse
import os

def load_image(image_path, device='cuda'):
    moda_a = np.load(image_path)
    if len(moda_a.shape) == 2:
        moda_a = moda_a[:, :]
    elif len(moda_a.shape) == 3:
        moda_a = moda_a[100, :, :]
    else:
        raise ValueError('Unsupported array shape!')

    transform_a = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        ToTensor()
    ])

    data_a = transform_a(moda_a.astype(np.float32)).unsqueeze(0)
    return data_a.to(device)

def load_generator(checkpoint_path):
    model = cyclegan.CycleGen()
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda()
    return model

def extract_identifier(file_path):
    # Extracts only the filename without extension
    base_name = os.path.basename(file_path)
    identifier = os.path.splitext(base_name)[0]
    return identifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FedMed-GAN on a specified CT image.")
    parser.add_argument("image_path", type=str, help="Path to the input CT image.")
    parser.add_argument("--output_dir", type=str, default="/home/user/FedMed-GAN-main/work_dir/ct_to_mri_output/", help="Directory to save output images.")
    args = parser.parse_args()

    # Ensure output directory exists or create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generator_path = '/home/user/FedMed-GAN-main/work_dir/federated/synthrad/Self-Attention/checkpoint/g_from_b_to_a/best_model_mri_ct_13.7941.pth'
    generator = load_generator(generator_path)
    image_tensor = load_image(args.image_path)
    input_image = image_tensor.squeeze().cpu().numpy()
    
    identifier = extract_identifier(args.image_path)

    # Correct filename creation
    input_image_path = os.path.join(args.output_dir, f"{identifier}-input-CT.jpg")
    plt.imshow(input_image, cmap='gray')
    plt.axis('off')
    plt.savefig(input_image_path)
    plt.show()

    # Generate and save the output image
    with torch.no_grad():
        output = generator(image_tensor)

    output_image = output.squeeze().cpu().numpy()
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    output_image_path = os.path.join(args.output_dir, f"{identifier}-output-MRI.jpg")
    plt.savefig(output_image_path)
    plt.show()
