cuda_device=$1
test_name=$2
dataset=$3
protocol=$4
representation=$5

if [ ! -d ./checkpoints/${test_name} ];then
    mkdir -p ./checkpoints/${test_name}
fi

if [ ! -d ./checkpoints_distill/${test_name} ]; then
    mkdir -p ./checkpoints_distill/${test_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python distill_just.py \
  --lr 0.01 \
  --batch-size 64 \
  --hico-k 32768 \
  --hico-t 0.07 \
  --hico-temp 1e-3 \
  --pretrained  ./checkpoints/${test_name}/checkpoint_0450.pth.tar \
  --checkpoint-path ./checkpoints_distill/${test_name}_hicok32768 \
  --epochs 451  --pre-dataset ${dataset} --protocol ${protocol} \
  --skeleton-representation ${representation} | tee -a ./checkpoints_distill/${test_name}/pretraining.log
