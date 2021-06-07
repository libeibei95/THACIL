CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase train
CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase test
CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase train --pos-strategy 2
CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase test
