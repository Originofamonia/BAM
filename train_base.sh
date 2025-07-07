#!/bin/sh
PARTITION=Segmentation

GPU_ID=0,1  # 0 is 3090, 1 is 3060 Ti
dataset=coco # pascal coco
exp_name=split0

arch=PSPNet
net=resnet50 # vgg resnet50

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_base.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${config} ${exp_dir}

echo ${arch}
echo ${config}
export OMP_NUM_THREADS=6  # or some value based on your CPU core count
# --nproc_per_node=2 means number of GPUs
CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node=2 --master_port=1234 train_base.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log