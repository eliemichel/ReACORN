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

from argparse import ArgumentParser
from datetime import datetime
import torch

parser = ArgumentParser(description=
'''Display info from a checkpoint created by run_acorn.py''')

# Basic arguments
parser.add_argument('checkpoint', type=str, help='filename of the checkpoint to display')

def main(args):
    checkpoint = torch.load(args.checkpoint)
    
    if "timestamp" in checkpoint:
        dt = datetime.fromtimestamp(checkpoint["timestamp"])
        checkpoint_datetime = dt.strftime("%m/%d/%Y, %H:%M:%S")
    else:
        checkpoint_datetime = "(unknown date time)"

    cmd_args = checkpoint.get("cmd_args", {})

    print(f"Checkpoint taken on {checkpoint_datetime}:")
    for k, v in cmd_args.items():
        print(f" - {k} = {v}")

    encoder_state = checkpoint["model_state_dict"]["encoder"]
    print(encoder_state['B'])

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)