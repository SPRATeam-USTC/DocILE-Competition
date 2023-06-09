cd /path/to/DocILE_submit/model

export model_name_or_path=pretrained_model/pretrain_docile_topk36
# export model_name_or_path=pretrained_model/pretrain_docile_topk60
export output_dir=experiments/docile_lir_merge
export gradient_accumulation_steps=1
export per_device_train_batch_size=4
export per_device_eval_batch_size=4
export dataloader_num_workers=1
export num_train_epochs=1000

python -m torch.distributed.launch --nproc_per_node=2 runner/graphdoc/finetune_doclie_ocr_data_lir_merge.py \
--model_name_or_path $model_name_or_path \
--output_dir $output_dir \
--do_train \
--save_strategy epoch \
--logging_strategy steps \
--logging_steps 30 \
--save_total_limit 1000 \
--num_train_epochs $num_train_epochs \
--gradient_accumulation_steps $gradient_accumulation_steps \
--per_device_train_batch_size $per_device_train_batch_size \
--dataloader_num_workers $dataloader_num_workers \
--warmup_ratio 0.1 \
--fp16


#top60, train for 550 epoch

# export NGPUS=2
# export NNODES=4

# export model_name_or_path=pretrained_model/pretrain_docile_topk60
# export output_dir=experiments/docile_lir_merge
# export gradient_accumulation_steps=1
# export per_device_train_batch_size=4
# export per_device_eval_batch_size=4
# export dataloader_num_workers=4
# export num_train_epochs=1000
# export master_port=10006

# if [[ $NNODES -gt 1 ]]; then
#     python -m torch.distributed.launch --nproc_per_node $NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
#         runner/graphdoc/finetune_doclie_ocr_data_lir_merge.py \
#         --model_name_or_path $model_name_or_path \
#         --output_dir $output_dir \
#         --do_train \
#         --save_strategy epoch \
#         --logging_strategy steps \
#         --logging_steps 50 \
#         --save_total_limit 1000 \
#         --num_train_epochs $num_train_epochs \
#         --gradient_accumulation_steps $gradient_accumulation_steps \
#         --per_device_train_batch_size $per_device_train_batch_size \
#         --dataloader_num_workers $dataloader_num_workers \
#         --warmup_ratio 0.1 \
#         --fp16
# else
# 	python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$master_port \
#         runner/graphdoc/finetune_doclie_ocr_data_lir_merge.py \
#         --model_name_or_path $model_name_or_path \
#         --output_dir $output_dir \
#         --do_train \
#         --save_strategy epoch \
#         --logging_strategy steps \
#         --logging_steps 30 \
#         --save_total_limit 1000 \
#         --num_train_epochs $num_train_epochs \
#         --gradient_accumulation_steps $gradient_accumulation_steps \
#         --per_device_train_batch_size $per_device_train_batch_size \
#         --dataloader_num_workers $dataloader_num_workers \
#         --warmup_ratio 0.1 \
#         --fp16
# fi

