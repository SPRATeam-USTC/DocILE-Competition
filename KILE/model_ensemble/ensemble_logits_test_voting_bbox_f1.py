import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse


import pandas as pd

class DocTypeVocab:
    key_words = []

    def __init__(self):
        self._words_ids_map = dict()
        self._ids_words_map = dict()

        for word_id, word in enumerate(self.key_words):
            self._words_ids_map[word] = word_id
            self._ids_words_map[word_id] = word
    
    def __len__(self):
        return len(self._words_ids_map)

    def word_to_id(self, word):
        return self._words_ids_map[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word) for word in words]

    def id_to_word(self, word_id):
        return self._ids_words_map[word_id]
    
    def ids_to_words(self, words_id):
        return [self.id_to_word(word_id) for word_id in words_id]


class DocliekieEntityVocab(DocTypeVocab):
    key_words = ["account_num", "amount_due", "amount_paid", "amount_total_gross", "amount_total_net", "amount_total_tax",\
        "bank_num", "bic", "currency_code_amount_due", "customer_billing_name", "customer_billing_address", "customer_delivery_name",\
        "customer_id", "customer_order_id", "customer_other_address", "customer_other_name", "customer_delivery_address",\
        "customer_registration_id", "customer_tax_id", "date_due", "date_issue", "document_id", "iban", "order_id", "payment_reference",\
        "payment_terms", "tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax", "vendor_address", "vendor_email",\
        "vendor_name", "vendor_order_id", "vendor_registration_id", "vendor_tax_id", "line_item_amount_gross", "line_item_amount_net",\
        "line_item_code", "line_item_currency", "line_item_date", "line_item_description", "line_item_discount_amount", "line_item_discount_rate",\
        "line_item_hts_number", "line_item_order_id", "line_item_person_name", "line_item_position", "line_item_quantity", "line_item_tax",\
        "line_item_tax_rate", "line_item_unit_price_gross", "line_item_unit_price_net", "line_item_units_of_measure", "line_item_weight", "other"]

class DoclieonlykieEntityVocab(DocTypeVocab):
    key_words = ["account_num", "amount_due", "amount_paid", "amount_total_gross", "amount_total_net", "amount_total_tax",\
        "bank_num", "bic", "currency_code_amount_due", "customer_billing_name", "customer_billing_address", "customer_delivery_name",\
        "customer_id", "customer_order_id", "customer_other_address", "customer_other_name", "customer_delivery_address",\
        "customer_registration_id", "customer_tax_id", "date_due", "date_issue", "document_id", "iban", "order_id", "payment_reference",\
        "payment_terms", "tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax", "vendor_address", "vendor_email",\
        "vendor_name", "vendor_order_id", "vendor_registration_id", "vendor_tax_id", "other"]

def cal_classify_metrics_with_pred(threshold, predictions_origin, labels_origin):
    labels = []
    predictions = []
    labels_tmp = labels_origin
    predictions_tmp = predictions_origin
    for i,item in enumerate(labels_tmp):
        if sum(item) > 0:
            labels.append(item)
            predictions.append(predictions_tmp[i])

    labels = np.array(labels)
    predictions = np.array(predictions)

    tp_tmp = np.sum(predictions * labels, axis=0)
    fp_tmp = np.sum(predictions * (1-labels), axis=0)
    fn_tmp = np.sum((1-predictions) * labels, axis=0)
    tp = np.delete(tp_tmp, [7,18,22])
    fp = np.delete(fp_tmp, [7,18,22])
    fn = np.delete(fn_tmp, [7,18,22])

    precision = np.nan_to_num(tp / (tp + fp))
    recall = np.nan_to_num(tp / (tp + fn))
    f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    macro_f1 = np.mean(f1)
    TP = np.sum(tp)
    FP = np.sum(fp)
    FN = np.sum(fn)
    micro_precision = np.nan_to_num(TP / (TP + FP))
    micro_recall = np.nan_to_num(TP / (TP + FN))
    micro_F1 = np.nan_to_num(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall))
    return(micro_F1, micro_precision, micro_recall)


def cal_classify_metrics(threshold, predictions_origin, labels_origin):
    labels = []
    predictions = []
    labels_tmp = labels_origin
    predictions_tmp = predictions_origin
    for i,item in enumerate(labels_tmp):
        if sum(item) > 0:
            labels.append(item)
            predictions.append(predictions_tmp[i])

    labels = np.array(labels)
    predictions = np.array(predictions)
    predictions[predictions > threshold] = 1
    predictions[predictions <= threshold] = 0

    tp_tmp = np.sum(predictions * labels, axis=0)
    fp_tmp = np.sum(predictions * (1-labels), axis=0)
    fn_tmp = np.sum((1-predictions) * labels, axis=0)
    tp = np.delete(tp_tmp, [7,18,22])
    fp = np.delete(fp_tmp, [7,18,22])
    fn = np.delete(fn_tmp, [7,18,22])

    precision = np.nan_to_num(tp / (tp + fp))
    recall = np.nan_to_num(tp / (tp + fn))
    f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    macro_f1 = np.mean(f1)
    TP = np.sum(tp)
    FP = np.sum(fp)
    FN = np.sum(fn)
    micro_precision = np.nan_to_num(TP / (TP + FP))
    micro_recall = np.nan_to_num(TP / (TP + FN))
    micro_F1 = np.nan_to_num(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall))
    return(micro_F1, micro_precision, micro_recall)

   

def select_best_step(step_num, metrics_step_thre_path):
    metrics_step_thre = json.load(open(metrics_step_thre_path, "r"))
    max_f1_th_lst = []
    max_f1_lst = []
    step_lst = list(metrics_step_thre.keys()) 
    for step, metrics in metrics_step_thre.items():
        thres_lst = list(metrics.keys())
        thres_lst = [float(x) for x in thres_lst]
        metric_lst = [value for value in metrics.values()]
        f1_lst= [x[0] for x in metric_lst]
        
        max_f1_th_lst.append(thres_lst[f1_lst.index(max(f1_lst))])
        max_f1_lst.append(max(f1_lst))
    sorted_lst = sorted(zip(max_f1_lst,max_f1_th_lst,step_lst), reverse=True)
    sorted_max_f1_lst, sorted_max_f1_th_lst, sorted_step_lst = zip(*sorted_lst)
    return(sorted_step_lst[:step_num], sorted_max_f1_lst[:step_num], sorted_max_f1_th_lst[:step_num])

def select_best_step_ap(step_num, ap_path):
    metrics_steps_aps = json.load(open(ap_path, "r"))

    max_steps_lst = list(metrics_steps_aps.keys())
    max_aps_lst = list(metrics_steps_aps.values())

    sorted_lst = sorted(zip(max_aps_lst, max_steps_lst), reverse=True)
    sorted_max_ap_lst, sorted_step_lst = zip(*sorted_lst)

    return(sorted_step_lst[:step_num], sorted_max_ap_lst[:step_num])


def visual_model_fusion_metrics(f1_lst, p_lst, r_lst, step_num_list, img_save_path):
    plt.figure()
    plt.plot(step_num_list, f1_lst, color='r', label='F1')
    plt.plot(step_num_list, p_lst, color='g',  label='P')
    plt.plot(step_num_list, r_lst, color='b', label='R')
    plt.legend()
    plt.xlabel('steps_num')   
    plt.ylabel('F1/P/R')

    max_f1_index = f1_lst.index(max(f1_lst))
    max_f1_th = step_num_list[max_f1_index]
    max_f1_y = f1_lst[max_f1_index]

    plt.text(max_f1_th, max_f1_y, f'({max_f1_th}, {max_f1_y})', ha='center', va='bottom',color='c')
    plt.savefig(img_save_path)

    return()


if __name__ == "__main__":
    threshold = 0.1
    step_num = 11  
    micro_F1_lst = []
    micro_precision_lst = []
    micro_recall_lst = []
    vocab = DoclieonlykieEntityVocab()
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--top36_logits_dir", type=str, default=None)
    parser.add_argument("--top60_logits_dir", type=str, default=None)
    parser.add_argument("--result_save_path", type=str, default=None)
    args = parser.parse_args()

    top36_logits_dir = args.top36_logits_dir
    top60_logits_dir = args.top60_logits_dir
    result_save_path = args.result_save_path


    

    classify_preds = []
    result = {}
    
    tmp = np.load(os.path.join(top36_logits_dir, os.listdir(top36_logits_dir)[0]))
    prediction_tmp = np.zeros_like(tmp)
    prediction_tmp = prediction_tmp[np.newaxis, :, :]
    logits = np.zeros_like(tmp)

    for step_top36, step_top60 in zip(os.listdir(top36_logits_dir), os.listdir(top60_logits_dir)):
    
        logits_step_top36 = np.load(os.path.join(top36_logits_dir, step_top36))
        logits_step_top60 = np.load(os.path.join(top60_logits_dir, step_top60))

        logits = logits + logits_step_top36 + logits_step_top60

        logits_step_top36[logits_step_top36 > threshold] = 1
        logits_step_top36[logits_step_top36 <= threshold] = 0
        logits_step_top60[logits_step_top60 > threshold] = 1
        logits_step_top60[logits_step_top60 <= threshold] = 0
        prediction_tmp = np.concatenate((prediction_tmp, logits_step_top36[np.newaxis, :, :]), axis=0)
        prediction_tmp = np.concatenate((prediction_tmp, logits_step_top60[np.newaxis, :, :]), axis=0)

    prediction_sum = np.sum(prediction_tmp, axis=0)
    prediction = np.where(prediction_sum >= step_num, 1, 0)
    logits /= (step_num*2)

    for x in prediction:
        if x.sum() > 0:
            classify_preds.append(vocab.ids_to_words(np.where(x==1)[0].tolist()))
        else:
            classify_preds.append(["other"])
    
    result["classify_preds"] = classify_preds
    np.save(os.path.join(result_save_path, "logits.npy"), logits)
    with open(os.path.join(result_save_path, "classify_preds.json"), "w") as f:
        json.dump(result, f)
