cd /path/to/DocILE_submit/KILE/post_process

# post-processing of the classification and merge results of the models
python process_result_split_instance.py \ 
--model_ouputs_dir /path/to/kie_merge \
--result_json kile_merge_result.json \
--document_folder_path /path/to/doc_pdfs


