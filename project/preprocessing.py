import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from celluloid import Camera
from argparse import ArgumentParser
from tqdm import tqdm
import os
import cv2


def visualize_ct_scan(img_path, label_path):
    ct_scan_volume = nib.load(img_path).get_fdata()
    label_volume = nib.load(label_path).get_fdata().astype(np.uint8)

    fig = plt.figure()
    camera = Camera(fig)
    for idx in range(ct_scan_volume.shape[-1]):
        plt.imshow(ct_scan_volume[:, :, idx], cmap='bone')
        mask = np.ma.masked_where(label_volume[:, :, idx] == 0, label_volume[:, :, idx])
        plt.imshow(mask, alpha=0.5, cmap='autumn')
        camera.snap()

    animation = camera.animate()
    return animation


def resize_and_save(data, mask, output_path_prefix, name, size, vgg_compatible=True):
    resized_data = cv2.resize(data, size).astype(np.float32)
    if vgg_compatible:
        resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    np.save(os.path.join(output_path_prefix, 'data', name), resized_data)
    np.save(os.path.join(output_path_prefix, 'mask', name), resized_mask)


def preprocess_ct_scan(img_path, label_path, output_path_prefix, f_name, size,
                       frames_to_skip=30, orientation=('L', 'A', 'S'), vgg_compatible=True, scaling_value=3071):
    scan_data = nib.load(img_path)
    ct_scan_volume = nib.load(img_path).get_fdata() / scaling_value
    label_volume = nib.load(label_path).get_fdata().astype(np.uint8)
    os.makedirs(os.path.join(output_path_prefix, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_path_prefix, 'mask'), exist_ok=True)
    for idx in range(frames_to_skip, ct_scan_volume.shape[-1]):
        if nib.aff2axcodes(scan_data.affine) == orientation:
            name = f'{f_name}_{idx}'
            resize_and_save(ct_scan_volume[:, :, idx], label_volume[:, :, idx], output_path_prefix, name, size,
                            vgg_compatible)
        else:
            print(f"{f_name} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")


def preprocess(input_data_dir, input_labels_dir, output_dir, frames_to_skip, size, orientation, vgg_compatible,
               val_split=0.1):
    valid_files = [scan_file for scan_file in os.listdir(input_data_dir) if scan_file.startswith('lung') and
                   scan_file.endswith('.nii.gz')]
    train_size = len(valid_files) * (1-val_split)
    for idx, scan_file in tqdm(enumerate(valid_files)):
        output_prefix = 'train' if idx < train_size else 'val'
        output_path_prefix = os.path.join(output_dir, output_prefix)
        img_path = os.path.join(input_data_dir, scan_file)
        label_path = os.path.join(input_labels_dir, scan_file)
        f_name = scan_file.split(".")[0]
        with ProcessPoolExecutor() as executor:
            executor.submit(preprocess_ct_scan, img_path, label_path, output_path_prefix, f_name, size, frames_to_skip,
                            orientation, vgg_compatible)


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--orientation', type=tuple, default=('L', 'A', 'S'))
    parser.add_argument('--input_data_dir', type=str)
    parser.add_argument('--input_labels_dir', type=str)
    parser.add_argument('--output_dir', type=str, default=os.getcwd())
    parser.add_argument('--vgg_compatible', type=bool, default=True)
    parser.add_argument('--frames_to_skip', type=int, default=30)
    parser.add_argument('--resize', type=tuple, default=(224, 224))
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    preprocess(args.input_data_dir, args.input_labels_dir, args.output_dir, args.frames_to_skip, args.resize,
               args.orientation, args.vgg_compatible, args.val_split)


if __name__ == '__main__':
    cli_main()