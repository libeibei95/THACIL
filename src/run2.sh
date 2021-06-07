CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase train --neg-strategy 2
CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase test
CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase train --pos-strategy 2 --neg-strategy 2
CUDA_VISIBLE_DEVICES=1 python launcher.py  --phase test
