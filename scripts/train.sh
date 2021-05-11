model=sampling_free/retinanet_R_50_FPN_1x

CUDA_VISIBLE_DEVICES=7 python tools/train_net.py \
    --config-file configs/$model\.yaml \
    OUTPUT_DIR outputs/$model
