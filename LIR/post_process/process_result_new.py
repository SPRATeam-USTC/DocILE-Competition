import os
import cv2
import glob
import tqdm
import json
import os
import re
from collections import Counter
import argparse



class DocTypeVocab:
    key_words = [
    'letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific report', \
        'scientific publication', 'specification', 'file folder', 'news article', \
            'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo', 'docbank' ]

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

class DoclieonlylirEntityVocab(DocTypeVocab):
    key_words = ["line_item_amount_gross", "line_item_amount_net", "line_item_code", "line_item_currency", "line_item_date", \
        "line_item_description", "line_item_discount_amount", "line_item_discount_rate",  "line_item_hts_number", "line_item_order_id", \
        "line_item_person_name", "line_item_position", "line_item_quantity", "line_item_tax", "line_item_tax_rate", \
        "line_item_unit_price_gross", "line_item_unit_price_net", "line_item_units_of_measure", "line_item_weight"]


def remove_non_numeric_prefix(s):
    match = re.match(r'\D*(\d.*)', s)
    if match:
        prefix_len = len(s) - len(match.group(1))
        return (prefix_len, match.group(1))
    else:
        return (0, s)

def over_distance(instance: list) -> bool:
    sorted_box_x_list = sorted(instance, key=lambda x: x[0])
    width_list = [x[2]-x[0] for x in sorted_box_x_list]
    x_distance_list = [m[0]-n[0] for m,n in zip(sorted_box_x_list[1:], sorted_box_x_list[:-1])]
    height_list = [y[3]-y[1] for y in sorted_box_x_list]
    y_distance_list = [m[1]-n[1] for m,n in zip(sorted_box_x_list[1:], sorted_box_x_list[:-1])]
    min_x_distance = min([abs(x) for x in x_distance_list])
    max_y_distance = max([abs(y) for y in y_distance_list])
    max_width = max(width_list)
    max_height = max(height_list)
    if min_x_distance > 1.5*max_width or max_y_distance > 2*max_height:
        return True
    else:
        return False

def over_x_distance(box_list):
    width_list = [x[2]-x[0] for x in box_list]
    x_distance_list = [m[0]-n[0] for m,n in zip(box_list[1:], box_list[:-1])]
    if min(x_distance_list) > 2*max(width_list):
        return True
    else:
        return False



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ouputs_dir", type=str, default=None)
    parser.add_argument("--result_json", type=str, default=None)
    parser.add_argument("--document_names_json", type=str, default=None)
    parser.add_argument("--ann_path", type=str, default=None)
    args = parser.parse_args()

    model_ouputs_dir = args.model_ouputs_dir
    result_json = args.result_json
    document_names_json = args.document_names_json
    ann_path = args.ann_path

    with open(document_names_json, "r")  as f:
        document_names = json.load(f)

    input_files = glob.glob(os.path.join(model_ouputs_dir, '*'))
    vocab = DoclieonlylirEntityVocab()
    output_lir_result = {}
    for document_id in tqdm.tqdm(document_names):
        with open(os.path.join(ann_path, f"{document_id}.json")) as f:
            ann_info = json.load(f)
        page_num = ann_info["metadata"]["page_count"]
       
        output_lir_result[document_id] = []
        item_num = 0
        item_num_pre = 0
        for page_id in range(page_num):
            for path in input_files:
                if f'{document_id}_{page_id}' == os.path.basename(path)[:-5]:
                    item_num += item_num_pre
                    with open(path, "r") as f:
                        model_output = json.load(f)

                    item_num_pre = len(model_output["pred_item_instances_idxes"])
                    for i,instance_polys in enumerate(model_output["pred_instances_polys_idxes"]):
                        sub_info = {}
                        text_list_tmp = [model_output["gt_info"]["contents"][j] for j in instance_polys]
                        box_list_tmp = [model_output["gt_info"]["polys"][j] for j in instance_polys]

                        box_class_list = [x for j in instance_polys for x in model_output["classify_preds"][j]]
                        sorted_box_list = sorted(box_list_tmp, key=lambda x: x[0]) 
                        sorted_text_list = [text_list_tmp[box_list_tmp.index(p)] for p in sorted_box_list]
                        instance_class_tmp =  max(box_class_list, key=Counter(box_class_list).get)
                        line_item_id_tmp = next((j for j, row in enumerate(model_output["pred_item_instances_idxes"]) if i in row), -1)
                        if line_item_id_tmp == -1:
                            continue
                        line_item_id_tmp = line_item_id_tmp + item_num
                            
                        box_tmp = [min([row[0] for row in sorted_box_list]), min([row[1] for row in sorted_box_list]),
                                            max([row[2] for row in sorted_box_list]), max([row[3] for row in sorted_box_list])]
                        text_tmp = "".join(sorted_text_list)
                        if len(text_tmp) == 0:
                            continue

                        if len(sorted_box_list) > 1 and over_x_distance(sorted_box_list):
                            for i, box in enumerate(sorted_box_list):
                                sub_info["page"] = int(os.path.basename(path)[-6])
                                sub_info["score"] = None
                                sub_info["line_item_id"] = line_item_id_tmp
                                sub_info["use_only_for_ap"] = False
                                sub_info["fieldtype"] = instance_class_tmp
                                sub_info["bbox"] = box
                                sub_info["text"] = sorted_text_list[i]
                                output_lir_result[document_id].append(sub_info)
                                sub_info = {}
                        elif ("line_item_units_of_measure" in box_class_list or "line_item_quantity" in box_class_list) and "spots".lower() in text_tmp.lower():
                                len_piece = (box_tmp[2] - box_tmp[0]) / len(text_tmp)
                                if len(text_tmp) > 5:
                                    box_split_list = [[box_tmp[2]-len_piece*5, box_tmp[1], box_tmp[2], box_tmp[3]], \
                                                    [box_tmp[0], box_tmp[1], box_tmp[2]-len_piece*5, box_tmp[3]]]
                                    class_split_list = ["line_item_units_of_measure", "line_item_quantity"] + [x for x in box_class_list if x != "line_item_units_of_measure"]
                                    for i, box in enumerate(box_split_list):
                                        sub_info["page"] = int(os.path.basename(path)[-6])
                                        sub_info["score"] = None
                                        sub_info["line_item_id"] = line_item_id_tmp
                                        sub_info["use_only_for_ap"] = False
                                        sub_info["fieldtype"] = class_split_list[i]
                                        sub_info["bbox"] = box
                                        sub_info["text"] = text_tmp
                                        output_lir_result[document_id].append(sub_info)
                                        sub_info = {}
                                else:
                                    sub_info["page"] = int(os.path.basename(path)[-6])
                                    sub_info["score"] = None
                                    sub_info["line_item_id"] = line_item_id_tmp
                                    sub_info["use_only_for_ap"] = False
                                    sub_info["fieldtype"] = instance_class_tmp
                                    sub_info["bbox"] = box_tmp
                                    sub_info["text"] = text_tmp
                                    output_lir_result[document_id].append(sub_info)
                                    sub_info = {}
                        
                        else:
                            sub_info["page"] = int(os.path.basename(path)[-6])
                            sub_info["score"] = None
                            sub_info["line_item_id"] = line_item_id_tmp
                            sub_info["use_only_for_ap"] = False
                            sub_info["fieldtype"] = instance_class_tmp
                            sub_info["bbox"] = box_tmp
                            sub_info["text"] = text_tmp
                            output_lir_result[document_id].append(sub_info)
                            sub_info = {}


    print(len(output_lir_result)) 
    with open(result_json, "w") as f:
        json.dump(output_lir_result, f)








