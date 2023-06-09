cd /path/to/KILE/val_test_dataprocess

# generate model input data for inference
python extract_infer_input_data.py \ 
--image_dir /path/to/imgs \
--ocr_dir /path/to/ocr_result_without_bigbox_ordered.json \
--output_json_path test/infos.json \
--output_embed_dir test/embeddings


