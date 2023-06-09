import os
import cv2
import glob
import tqdm
import json
import os
import re
import argparse



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ouputs_dir", type=str, default=None)
    parser.add_argument("--result_json", type=str, default=None)
    parser.add_argument("--document_folder_path", type=str, default=None)
    args = parser.parse_args()

    model_ouputs_dir = args.model_ouputs_dir
    result_json = args.result_json
    document_folder_path = args.document_folder_path
   
    document_names = [f[:-4] for f in os.listdir(document_folder_path) if os.path.isfile(os.path.join(document_folder_path, f))]
    input_files = glob.glob(os.path.join(model_ouputs_dir, '*'))
    vocab = DocliekieEntityVocab()
    output_kile_result = {}
    for document_id in document_names:
        output_kile_result[document_id] = []
        for path in input_files:
            if document_id in path:
                with open(path, "r") as f:
                    model_output = json.load(f)
                for i, instance_class in enumerate(model_output["merger_preds_classes"]):
                    sub_info = {}
                    text_list_tmp = [model_output["gt_info"]["contents"][j] for j in model_output["merger_preds_instances"][i]]
                    box_list_tmp = [model_output["gt_info"]["polys"][j] for j in model_output["merger_preds_instances"][i]]
                    instance_classify_score_list = [model_output["classify_logits"][j][vocab.word_to_id(instance_class)] for j in model_output["merger_preds_instances"][i]]
                    if "name" not in instance_class and "address" not in instance_class and len(box_list_tmp) > 1 and over_distance(box_list_tmp):
                        for text_i,box_i,instance_classify_score in zip(text_list_tmp, box_list_tmp, instance_classify_score_list):
                            if len(text_i) == 0:
                                continue
                            if instance_class == "currency_code_amount_due":
                                if "$" in text_i:
                                    pos = text_i.index("$") 
                                    split_s = (box_i[2]-box_i[0])/len(text_i)
                                    box = [split_s*pos+box_i[0], box_i[1], split_s*(pos+1)+box_i[0], box_i[3]]
                                    text = "$"
                                else:
                                    box = [box_i[0], box_i[1], (box_i[2]-box_i[0])/len(text_i) + box_i[0], box_i[3]]
                                    text = text_i[0]
                            elif "id" in instance_class and text_i.startswith("#"):
                                box = [(box_i[2]-box_i[0])/len(text_i) + box_i[0], box_i[1], box_i[2], box_i[3]]
                                text = text_i[1:]
                            else:
                                box = box_i
                                text = text_i
                                
                            sub_info["page"] = int(os.path.basename(path)[-6])
                            # sub_info["score"] = None
                            # sub_info["score"] = instance_classify_score
                            sub_info["score"] = (instance_classify_score + model_output["merger_preds_scores"][i])/2
                            # sub_info["score"] = model_output["merger_preds_scores"][i]
                            sub_info["line_item_id"] = None
                            sub_info["use_only_for_ap"] = False
                            sub_info["fieldtype"] = instance_class
                            sub_info["bbox"] = box
                            sub_info["text"] = text
                            output_kile_result[document_id].append(sub_info)
                            sub_info = {}

                    else:
                        instance_classify_score = sum(instance_classify_score_list) / len(instance_classify_score_list)
                        sorted_box_list = sorted(box_list_tmp, key=lambda x: x[0]) 
                        sorted_text_list = [text_list_tmp[box_list_tmp.index(p)] for p in sorted_box_list]
                        box_tmp = [min([row[0] for row in sorted_box_list]), min([row[1] for row in sorted_box_list]),
                                        max([row[2] for row in sorted_box_list]), max([row[3] for row in sorted_box_list])]
                        text_tmp = " ".join(sorted_text_list)
                        if len(text_tmp) == 0:
                            continue
                        if instance_class == "currency_code_amount_due":
                            if "$" in text_tmp:
                                pos = text_tmp.index("$") 
                                split_s = (box_tmp[2]-box_tmp[0])/len(text_tmp)
                                box = [split_s*pos+box_tmp[0], box_tmp[1], split_s*(pos+1)+box_tmp[0], box_tmp[3]]
                                text = "$"
                            else:
                                box = [box_tmp[0], box_tmp[1], (box_tmp[2]-box_tmp[0])/len(text_tmp) + box_tmp[0], box_tmp[3]]
                                text = text_tmp[0]
                        elif "id" in instance_class and text_tmp.startswith("#"):
                            box = [(box_tmp[2]-box_tmp[0])/len(text_tmp) + box_tmp[0], box_tmp[1], box_tmp[2], box_tmp[3]]
                            text = text_tmp[1:]
                        else:
                            box = box_tmp
                            text = text_tmp

                        sub_info["page"] = int(os.path.basename(path)[-6])
                        # sub_info["score"] = None
                        # sub_info["score"] = instance_classify_score
                        sub_info["score"] = (instance_classify_score + model_output["merger_preds_scores"][i])/2
                        # sub_info["score"] = model_output["merger_preds_scores"][i]
                        sub_info["line_item_id"] = None
                        sub_info["use_only_for_ap"] = False
                        sub_info["fieldtype"] = instance_class
                        sub_info["bbox"] = box
                        sub_info["text"] = text
                        output_kile_result[document_id].append(sub_info)
                        sub_info = {}

    print(len(output_kile_result)) 
    with open(result_json, "w") as f:
        json.dump(output_kile_result, f)








                
    