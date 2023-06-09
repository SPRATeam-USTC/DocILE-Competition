cd /path/to/DocILE_submit/KILE/train_dataprocess

# extract the required information from the official annotations and process it into the desired format for the task at hand
python ann_label.py \ 
--output_label_txt train_annlabel_kie.json \
--ann_path /path/to/train_ann

# generate training data and assign categories to each text box
python extract_kie_merge_npy.py \
--image_dir /path/to/train_img \
--ann_txt train_annlabel_kie.json \
--ocr_dir /path/to/ocr_result_without_bigbox_ordered.json \
--output_npy_path train/infos.npy \
--output_embed_dir train/embeddings

# group training data to address memory issues
python split500_kie_merge.py \
--loader_path train/infos.npy \
--output_npy_path train_split/infos.npy \
--output_embed_dir train_split/embeddings

