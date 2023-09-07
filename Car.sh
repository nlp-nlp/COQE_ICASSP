#!/bin/bash

for num in 60 
do
    for seed in 321
    do
        CUDA_VISIBLE_DEVICES=1 python main_sub_absa.py  \
        --model_name=Car_tuple \
        --num_generated $num \
        --decoder_lr=0.00003 \
        --encoder_lr=0.00001 \
        --data_path=data/Car-COQE \
        --bert_directory=/home/qtxu/PLM/bert-base-chinese \
        --batch_size=4 \
        --na_rel_coef=0.8 \
        --num_decoder_layers=3 \
        --max_epoch=20 \
        --max_grad_norm=10 \
        --random_seed $seed \
        --weight_decay=0.000001 \
        --lr_decay=0.02 \
        --max_text_length=512 \
        --kind=sub-absa \

    done
done


for num in 60 
do
    for seed in 321
    do
        CUDA_VISIBLE_DEVICES=1 python main_sub_absa.py  \
        --model_name=Car_quintule \
        --num_generated $num \
        --decoder_lr=0.00004 \
        --encoder_lr=0.00001 \
        --data_path=data/Car-COQE \
        --bert_directory=/home/qtxu/PLM/bert-base-chinese \
        --batch_size=4 \
        --na_rel_coef=0.5 \
        --num_decoder_layers=3 \
        --max_epoch=50 \
        --max_grad_norm=10 \
        --random_seed $seed \
        --weight_decay=0.000001 \
        --lr_decay=0.02 \
        --max_text_length=512 \
        --kind=main \
        --pt_file=first-stage-save-path/best.pt \

    done
done
