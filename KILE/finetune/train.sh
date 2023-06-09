cd /path/to/DocILE_submit/model


export model_name_or_path=pretrained_model/pretrain_docile_topk36
# export model_name_or_path=pretrained_model/pretrain_docile_topk60
export output_dir=experiments/docile_kie_merge
export gradient_accumulation_steps=1
export per_device_train_batch_size=4
export per_device_eval_batch_size=4
export dataloader_num_workers=1
export num_train_epochs=300

python -m torch.distributed.launch --nproc_per_node=2 runner/graphdoc/finetune_doclie_ocr_data_kie_merge36.py \
--model_name_or_path $model_name_or_path \
--output_dir $output_dir \
--do_train \
--save_strategy epoch \
--logging_strategy steps \
--logging_steps 30 \
--save_total_limit 300 \
--num_train_epochs $num_train_epochs \
--gradient_accumulation_steps $gradient_accumulation_steps \
--per_device_train_batch_size $per_device_train_batch_size \
--dataloader_num_workers $dataloader_num_workers \
--warmup_ratio 0.1 \
--fp16
