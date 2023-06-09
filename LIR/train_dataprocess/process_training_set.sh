cd /path/to/DocILE_submit/LIR/train_dataprocess

# extract the required information from the official annotations and process it into the desired format for the task at hand
python ann_label.py \ 
--output_label_txt train_annlabel_lir.json \
--ann_path /path/to/train_ann

# generate training data and assign categories to each text box
python extract_lir_merge_npy.py \
--image_dir /path/to/train_img \
--ann_txt train_annlabel_lir.json \
--ocr_dir /path/to/ocr_result_without_bigbox_ordered.json \
--output_npy_path train/infos.npy \
--output_embed_dir train/embeddings

# group training data to address memory issues
python split500_lir_merge_npy.py \
--info_origin_result train/infos.npy \
--output_npy_path train_split/infos.npy \
--output_embed_dir train_split/embeddings

