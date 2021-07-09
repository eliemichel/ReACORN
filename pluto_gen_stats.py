# This file is part of ReACORN, a reimplementation by Élie Michel of the ACORN
# paper by Martel et al. published at SIGGRAPH 2021.
#
# Copyright (c) 2021 -- Télécom Paris (Élie Michel <elie.michel@telecom-paris.fr>)
# 
# The MIT license:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided “as is”, without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and non-infringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising
# from, out of or in connection with the software or the use or other dealings
# in the Software.

import torch
import os
from argparse import Namespace
from matplotlib.image import imread, imsave
from matplotlib import pyplot as plt
import numpy as np

import acorn_eval_image

target_image_filname = "data/pluto.png"
checkpoint_dir = "checkpoints/pluto"
output_dir = "outputs/pluto"
device = "cuda"  # 'cpu' or 'cuda'

quadtree_output_dir = os.path.join(output_dir, "quadtree")
image_output_dir = os.path.join(output_dir, "outputs")
difference_output_dir = os.path.join(output_dir, "difference")
loss_output_dir = os.path.join(output_dir, "loss_plot")
psnr_output_dir = os.path.join(output_dir, "psnr_plot")
os.makedirs(quadtree_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(difference_output_dir, exist_ok=True)
os.makedirs(loss_output_dir, exist_ok=True)
os.makedirs(psnr_output_dir, exist_ok=True)

def main():
    gen_quadtree_images()
    gen_output_images()
    measure_differences()
    gen_loss_plots()
    gen_psnr_plots()

def gen_quadtree_images():
    print("Generating quadtree images...")
    for checkpoint in os.listdir(checkpoint_dir):
        name, _ = os.path.splitext(checkpoint)

        output_file = os.path.join(quadtree_output_dir, name + ".png")
        if os.path.exists(output_file):
            continue
        
        acorn_eval_image.main(Namespace(
            checkpoint = os.path.join(checkpoint_dir, checkpoint),
            image = output_file,
            resolution = "2048x2048",
            draw_quadtree = True,
            draw_quadtree_only = True,
            device = device,
        ))

def gen_output_images():
    print("Generating full res output images...")
    for checkpoint in os.listdir(checkpoint_dir):
        name, _ = os.path.splitext(checkpoint)

        output_file = os.path.join(image_output_dir, name + ".png")
        if os.path.exists(output_file):
            continue
        
        acorn_eval_image.main(Namespace(
            checkpoint = os.path.join(checkpoint_dir, checkpoint),
            image = output_file,
            resolution = "4096x4096",
            draw_quadtree = False,
            draw_quadtree_only = False,
            device = device,
        ))

def measure_differences():
    print("Measuring difference to ground truth...")
    target_image = imread(target_image_filname)
    if target_image.dtype == np.uint8:
        target_image = target_image.astype(float) / 255.

    for output_image_filename in os.listdir(image_output_dir):
        name, _ = os.path.splitext(output_image_filename)

        diff_filename = os.path.join(difference_output_dir, name + ".png")
        psnr_filename = os.path.join(difference_output_dir, name + ".txt")
        if os.path.exists(diff_filename):
            continue

        print(output_image_filename)
        output_image = imread(os.path.join(image_output_dir, output_image_filename))[:,:,:3]
        mse = np.power(output_image - target_image, 2).mean()
        psnr = 20 * np.log10(1 / np.sqrt(mse))
        with open(psnr_filename, 'w') as f:
            f.write(f"psnr={psnr}")
        print(f"psnr={psnr}")

        diff_image = np.ones_like(output_image)
        diff = np.abs(output_image - target_image).mean(axis=-1)
        diff_image[:,:,0] = 1
        diff_image[:,:,1] = (1 - diff).clip(0, 1)
        diff_image[:,:,2] = (1 - diff).clip(0, 1)
        imsave(diff_filename, diff_image)

def gen_loss_plots(size=(1152,256)):
    print("Generating loss plots...")
    last_checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[-1])
    max_epochs = int(last_checkpoint.split('.')[-2])

    for checkpoint_filename in os.listdir(checkpoint_dir):
        name, _ = os.path.splitext(checkpoint_filename)

        output_file = os.path.join(loss_output_dir, name + ".png")
        if os.path.exists(output_file):
            continue

        print(name)
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_filename))
        loss_log = checkpoint['loss_log']

        dpi = 96
        fig, ax = plt.subplots()
        fig.set_size_inches(size[0]/dpi, size[1]/dpi)
        fig.patch.set_visible(False)
        #ax.axis('off')

        ax.plot(loss_log)
        ax.set_xlim(-max_epochs*.01, max_epochs*1.01)
        ax.set_ylim(-0.005, 0.18)
        fig.savefig(output_file, transparent=True, dpi=dpi)
        plt.close(fig)

def gen_psnr_plots(size=(550,256)):
    print("Generating PSNR plots...")
    last_checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[-1])
    max_epochs = int(last_checkpoint.split('.')[-2])

    psnr_log = []
    epochs = []

    for i, filename in enumerate(os.listdir(difference_output_dir)):
        name, ext = os.path.splitext(filename)
        if ext != '.txt':
            continue

        output_file = os.path.join(psnr_output_dir, name + ".png")
        if os.path.exists(output_file):
            continue

        print(name)
        with open(os.path.join(difference_output_dir, filename)) as f:
            psnr = float(f.read().split("=")[-1])
        psnr_log.append(psnr)
        epochs.append(50 * i)

        dpi = 96
        fig, ax = plt.subplots()
        fig.set_size_inches(size[0]/dpi, size[1]/dpi)
        fig.patch.set_visible(False)
        #ax.axis('off')
        ax.get_xaxis().set_ticks([])

        ax.plot(epochs, psnr_log)
        ax.set_xlim(-max_epochs*.01, max_epochs*1.01)
        ax.set_ylim(0, 30)
        fig.savefig(output_file, transparent=True, dpi=dpi)
        plt.close(fig)

main()
