source activate YourOwnEnv

export start_index=307060
export end_index=318060
export process_id=1

# line level
export src_pdf_path=/path/to/pdf # folder that save pdf files in .pdf format
export src_ocr_path=/path/to/ocr  # folder that save ocr files in .json format
export pdf_list=/path/to/all_pdf_name.json  # a list that save valid pdf file name in .json format
export tgt_image_path=/path/to/images  # folder that save pdf-rendered image files in .png format
export output_info_dir=/path/to/subprocess_ocr_line_${process_id}  # folder that save current sub-process program output infos

# process ocr information into line-level ocr text and boxes, filter out low-quality documents
python /path/to/process_ocr_line.py \
        --src_pdf_path          $src_pdf_path       \
        --src_ocr_path          $src_ocr_path       \
        --pdf_list              $pdf_list           \
        --tgt_image_path        $tgt_image_path     \
        --output_info_dir       $output_info_dir     \
        --st_idx                $start_index        \
        --ed_idx                $end_index 