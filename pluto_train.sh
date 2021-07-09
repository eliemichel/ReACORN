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

# This reproduces the setup described in the paper's appendix
# regarding the Pluto image.
# It is recommended to run this file manually command by command.

mkdir -p checkpoints
mkdir -p data
mkdir -p outputs

# Download Pluto image
wget http://pluto.jhuapl.edu/Galleries/Featured-Images/pics/BIG_P_COLOR_2_TRUE_COLOR1.png -o data/pluto8k.png

# Rescale it to 4096x4096 using image magick, you may also do it manually in any other tool
magick convert data/pluto8k.png -resize 4096x4096 data/pluto.png

# Training
# (use --end_checkpoint checkpoints/pluto/pluto.######.tar to keep all intermediary checkpoints)
python acorn_train.py \
    data/pluto.png \
    --end_checkpoint checkpoints/pluto.tar \
    --device cuda \
    \
    --encoder_hidden_layer_count 4 \
    --encoder_hidden_layer_size 512 \
    --feature_size 16 \
    --feature_grid_resolution 32 \
    --fourier_feature_size 6 \
    --decoder_hidden_layer_size 64 \
    --max_active_blocks 1024 \
    \
    --epochs 100000 \
    --learning_rate 1e-3 \
    --optimize_partition_period 500 \
    --alpha 0.2 \
    --beta 0.02 \
    \
    --resample_period 1 \
    --save_checkpoint_period 50 \

# Testing
python acorn_eval_image.py \
    checkpoints/pluto.tar \
    outputs/pluto.png \
    1024x1024

# Generating figure data
# This scripts expects that there are intermediate checkpoint saved by replacing
# --end_checkpoint checkpoints/pluto.tar
# by
# --end_checkpoint checkpoints/pluto/pluto.######.tar
# in the training line above.
python pluto_gen_stats.py
