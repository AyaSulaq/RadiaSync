import os
import shutil

# Define the base path for the dataset
base_path = '/home/user/FedMed-GAN-main/Dataset/'

# Define the destination path for MRI images
destination_path = os.path.join(base_path, 'MRI')


# Iterate over each volume directory within the dataset
for volume_name in os.listdir(base_path):
    volume_path = os.path.join(base_path, volume_name)

    # Skip files, process only directories
    if not os.path.isdir(volume_path):
        continue

    # Define the path to the MRI image within the current volume directory
    mri_image_path = os.path.join(volume_path, 'mr.nii.gz')

    # Check if the MRI image exists
    if os.path.exists(mri_image_path):
        # Define the new file name and path for the MRI image
        new_file_name = f'{volume_name}.nii.gz'
        new_file_path = os.path.join(destination_path, new_file_name)

        # Move and rename the MRI image
        shutil.copy(mri_image_path, new_file_path)
        print(f'Moved and renamed: {mri_image_path} to {new_file_path}')
