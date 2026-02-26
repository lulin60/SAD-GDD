

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  train.py \
--cfg configs/moby_swin_base.yaml  --batch-size-train 14 --batch-size-test 20 \
--ad True --coupling_layers 8 --feats_l 1 --ad_dim 512 --ce_alpha 1.0 \
--bgspp_lambda 1.0 --contrast_alpha 0.018 \
--model_name swin_ad2stream \
