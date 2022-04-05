from argparse import ArgumentParser
import os

import cv2
import torch
import numpy as np
import nibabel as nib

from project.models.lit_segmentation_model import LitLungTumorSegModel
from project.models.segnet import *


def ct_slices_generator(img_path, size=(224, 224), orientation=('L', 'A', 'S'), vgg_compatible=True,
                        scaling_value=3071):
    scan_data = nib.load(img_path)
    ct_scan_volume = nib.load(img_path).get_fdata() / scaling_value
    for idx in range(ct_scan_volume.shape[-1]):
        if nib.aff2axcodes(scan_data.affine) == orientation:
            original_shape = ct_scan_volume[:, :, idx].shape
            resized_data = cv2.resize(ct_scan_volume[:, :, idx], size).astype(np.float32)
            if vgg_compatible:
                resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
            yield np.moveaxis(resized_data, -1, 0), original_shape
        else:
            print(f"{img_path} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")

'''
Inference saves a nifty file mask and masks for each slice
'''


def infer(path_to_ckpt, ct_slices, path_to_result_dir, name):
    model = LitLungTumorSegModel.load_from_checkpoint(path_to_ckpt)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    results = []
    for idx, (scan_data, original_shape) in enumerate(ct_slices):
        with torch.no_grad():
            mask = model(torch.from_numpy(np.expand_dims(scan_data, axis=0)).to(device))
            resized_mask = cv2.resize(mask.squeeze(0).cpu().numpy(), original_shape, interpolation=cv2.INTER_NEAREST)
            results.append(resized_mask)
        np.save(os.path.join(path_to_result_dir, f"{idx}_{name}"), resized_mask)
    full_mask = np.stack(results, axis=-1)
    nifti_mask = nib.Nifti1Image(full_mask, affine=np.eye(4))
    nib.save(nifti_mask, os.path.join(path_to_result_dir, f"label_{name}"))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--path_to_ckpt', type=str)
    parser.add_argument('--path_to_ct_scan', type=str)
    parser.add_argument('--path_to_result_dir', type=str, default=os.getcwd())
    parser.add_argument('--vgg_compatible', type=bool, default=True)
    parser.add_argument('--resize', type=tuple, default=(224, 224))
    parser.add_argument('--orientation', type=tuple, default=('L', 'A', 'S'))
    args = parser.parse_args()

    name = os.path.basename(args.path_to_ct_scan)
    preprocessed_ct_scan = ct_slices_generator(args.path_to_ct_scan, args.resize, args.orientation, args.vgg_compatible)
    infer(args.path_to_ckpt, preprocessed_ct_scan, args.path_to_result_dir, name)


if __name__ == '__main__':
    cli_main()
