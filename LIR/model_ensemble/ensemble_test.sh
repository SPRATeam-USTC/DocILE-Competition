cd /path/to/DocILE_submit/LIR/model_ensemble

# obtain classification results for the ensemble model based on the classification results of the selected models used for model ensemble
python ensemble_logits_test_averaging_official_metric.py \
--top36_logits_dir /path/to/top36_logits \
--top60_logits_dir /path/to/top60_logits \
--result_save_path /path/to/lir_top36_top60_fusion

cd /path/to/DocILE_submit/model

# inference process after model ensemble
python runner/graphdoc/docile_infer_fusion_lir.py \
--input_json /path/to/test/infos.json \
--checkpoint experiments/docile_lir_merge/checkpoint \
--output_dir /path/to/lir_top36_top60_fusion/lir_merge_fusion \
--classify_pred /path/to/lir_top36_top60_fusion/classify_preds.json \
--logits_path /path/to/lir_top36_top60_fusion/logits.npy


