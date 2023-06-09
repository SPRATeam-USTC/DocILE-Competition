cd /path/to/ocr_dataprocess

# convert the detection and recognition results from Doctr into the desired format
python scripts/ocr_label.py \ 
--output_label_txt ocr_official_snap.json \
--ocr_path /path/to/ocr_official_snap \
--geometry snapped_geometry

python scripts/ocr_label.py \ 
--output_label_txt ocr_1.25.json \
--ocr_path /path/to/ocr_1.25 \
--geometry geometry


python scripts/ocr_label.py \ 
--output_label_txt ocr_1.5.json \
--ocr_path /path/to/ocr_1.5 \
--geometry geometry


python scripts/ocr_label.py \ 
--output_label_txt ocr_1.75.json \
--ocr_path /path/to/ocr_1.75 \
--geometry geometry

# process OCR results obtained from three different sizes (similar to NMS)
python scripts/nms.py \ 
--img_path /path/to/imgs \
--ocrlabel_nms ocr_nms.json \
--label_125 ocr_1.25.json \
--label_15 ocr_1.5.json \
--label_175 ocr_1.75.json 

# combine OCR results obtained from three different sizes with the official OCR results
python scripts/combine_nmsocr_professocr.py \ 
--img_path /path/to/imgs \
--nms_ocr_path ocr_nms.json \
--profess_snap_ocr_path ocr_official_snap.json\
--ocr_result_path ocr_result.json

# remove large text boxes generated by errors during the detection process
python scripts/del_ocr_bigbox.py \ 
--img_path /path/to/imgs \
--ocr_origin_path ocr_result.json \
--ocr_result_path ocr_result_without_bigbox.json

# sort text boxes from left to right and from top to bottom.
python scripts/order_ocrlabel.py \ 
--ocr_label ocr_result_without_bigbox.json \
--ocr_label_order ocr_result_without_bigbox_ordered.json

