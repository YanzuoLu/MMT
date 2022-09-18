#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
CLUSTER=$4
GPU=$5

if [ $# -ne 5 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <CLUSTER NUM> <GPU>"
    exit 1
fi

export OMP_NUM_THREADS=32

# CUDA_VISIBLE_DEVICES=${GPU} \
# python mmt_train_kmeans.py -dt ${TARGET} -a ${ARCH} --num-clusters ${CLUSTER} \
# 	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 \
# 	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 \
# 	--print-freq 10 --eval-step 4 --workers 32 \
# 	--init-1 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-empty-1/model_best.pth.tar \
# 	--init-2 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-empty-2/model_best.pth.tar \
# 	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-MMT-empty-test-${CLUSTER}

CUDA_VISIBLE_DEVICES=${GPU} \
python mmt_train_kmeans.py -dt ${TARGET} -a ${ARCH} --num-clusters ${CLUSTER} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 \
	--print-freq 10 --eval-step 1 --workers 32 \
	--init /home/lyz/documents/OUDA/runs/mmt/duke2msmt/ckpt_source_pretrained.pth \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-MMT-init-${CLUSTER}
