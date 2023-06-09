cd /path/to/DocILE_submit/LIR/post_process

# post-processing of the classification and merge results of the models
python process_result_new.py \ 
--model_ouputs_dir /path/to/lir_merge \
--result_json lir_merge_result.json \
--document_names_json /path/to/test.json \
--ann_path /path/to/test_ann


