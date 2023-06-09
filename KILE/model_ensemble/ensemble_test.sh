cd /path/to/DocILE_submit/KILE/model_ensemble

# obtain classification results for the ensemble model based on the classification results of the selected models used for model ensemble
python ensemble_logits_test_voting_bbox_f1.py \
--top36_logits_dir /path/to/top36_logits \
--top60_logits_dir /path/to/top60_logits \
--result_save_path /path/to/kie_top36_top60_fusion

cd /path/to/DocILE_submit/model

# inference process after model ensemble
python runner/graphdoc/docile_infer_fusion.py \
--input_json /path/to/test/infos.json \
--checkpoint experiments/docile_kie_merge/checkpoint \
--output_dir /path/to/kie_top36_top60_fusion/kie_merge_fusion \
--classify_pred /path/to/kie_top36_top60_fusion/classify_preds.json \
--logits_path /path/to/kie_top36_top60_fusion/logits.npy


