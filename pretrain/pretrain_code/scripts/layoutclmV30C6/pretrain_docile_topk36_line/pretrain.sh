cd ../../../runner/layoutclmV30
source activate YourOwnEnv

export NGPUS=8
export NNODES=1

export model_name_or_path=../../libs/configs/layoutclmV30C6_topk36
export output_dir=../../experiments/layoutclmV30C6/pretrain_docile_topk36_line
export gradient_accumulation_steps=1
export per_device_train_batch_size=30
export per_device_eval_batch_size=30
export dataloader_num_workers=4
export num_train_epochs=3
export learning_rate=5e-5
export master_port=10006

if [[ $NNODES -gt 1 ]]; then
    python -m torch.distributed.launch --nproc_per_node $NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        pretrain.py \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy epoch \
        --logging_strategy steps \
        --logging_steps 50 \
        --label_names dtc_labels bdp_labels \
        --num_train_epochs $num_train_epochs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio 0.1 \
        --learning_rate $learning_rate \
        --ignore_data_skip True \
        --fp16
else
	python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$master_port \
        pretrain.py \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy epoch \
        --logging_strategy steps \
        --logging_steps 50 \
        --label_names dtc_labels bdp_labels \
        --num_train_epochs $num_train_epochs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio 0.1 \
        --learning_rate $learning_rate \
        --ignore_data_skip True \
        --fp16
fi