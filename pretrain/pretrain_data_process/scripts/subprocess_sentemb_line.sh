source activate YourOwnEnv

export start_index=0
export end_index=30000
export jobid=1

# line level
export ocr_path=/path/to/all_ocr.npy # merged list of ocr information saved in .npy format
export output_info_dir=/path/to/info_${jobid}/ # folder to save this part of processed sentence embedding paths info
export output_embed_dir=/path/to/sentence_embeddings/ # folder to save all sentence embeddings of text lines in pdf-rendered pages

# extract sentence embeddings and save path information
python process_sentence_embedding.py \
        --ocr_path          $ocr_path       \
        --output_info_dir   $output_info_dir \
        --output_embed_dir  $output_embed_dir \
        --start_index       $start_index    \
        --end_index         $end_index