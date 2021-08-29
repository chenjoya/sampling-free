model=sampling_free/atss_R_50_FPN_1x
vi outputs/$model/config\.yaml

gpus=4,5
gpun=2

for iter in "0090000"
do
    weight=outputs/$model/model_$iter\.pth
    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$gpun \
    tools/test.py --config-file outputs/$model/config\.yaml MODEL.WEIGHT $weight
done 
