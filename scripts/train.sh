config_file="configs/sampling_free/atss_R_50_FPN_1x.yaml"
gpus=4,5
gpun=2

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$gpun \
    tools/train.py --config-file $config_file
