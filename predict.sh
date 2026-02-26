#!/bin/bash

python predict.py \
--ckpt_name b14ceAlpha1.0_bgspp1.0_ct0.018_posebeta0.01_lr5_Clipgrad1.0_rgbdctHf_ADTrue_Clayers8_dim512 \
--test_flag video --test_data ff_c23 \
--model_name swin_ad2stream \
--cfg configs/moby_swin_base.yaml  --batch-size-test 100 \
--ad True --coupling_layers 8 --feats_l 1 --ad_dim 512 \

printf "\n"

