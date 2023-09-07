#!/bin/bash
for layer in 3
do
    for seed in 111
    do
        CUDA_VISIBLE_DEVICES=1 python main_sub_absa.py  \
        --model_name=Camera_tuple\
        --num_generated=60 \
        --decoder_lr=0.00005 \
        --encoder_lr=0.00002 \
        --data_path=data/Camera-COQE \
        --bert_directory=/home/qtxu/PLM/bert-base-uncased \
        --batch_size=4 \
        --na_rel_coef=0.2 \
        --num_decoder_layers $layer \
        --max_epoch=15 \
        --max_grad_norm=10 \
        --random_seed $seed \
        --weight_decay=0.000001 \
        --lr_decay=0.02 \
        --max_text_length=512 \
        --kind=sub-absa \
        
    done
done

for layer in 3
do
    for seed in 111
    do
        CUDA_VISIBLE_DEVICES=0 python main_sub_absa.py  \
        --model_name=Camera_quintuple \
        --num_generated=60 \
        --decoder_lr=0.00003 \
        --encoder_lr=0.00002 \
        --data_path=data/Camera-COQE \
        --bert_directory=/home/qtxu/PLM/bert-base-uncased \
        --batch_size=4 \
        --na_rel_coef=0.2 \
        --num_decoder_layers $layer \
        --max_epoch=15 \
        --max_grad_norm=10 \
        --random_seed $seed \
        --weight_decay=0.000001 \
        --lr_decay=0.02 \
        --max_text_length=512 \
        --kind=main \
        --pt_file=first-stage-save-path/best.pt \
        
    done
done
