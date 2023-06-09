import numpy as np


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
        "line_item_tax_rate", "line_item_unit_price_gross", "line_item_unit_price_net", "line_item_units_of_measure", "line_item_weight"]


class DoclieonlykieEntityVocab(DocTypeVocab):
    key_words = ["account_num", "amount_due", "amount_paid", "amount_total_gross", "amount_total_net", "amount_total_tax",\
        "bank_num", "bic", "currency_code_amount_due", "customer_billing_name", "customer_billing_address", "customer_delivery_name",\
        "customer_id", "customer_order_id", "customer_other_address", "customer_other_name", "customer_delivery_address",\
        "customer_registration_id", "customer_tax_id", "date_due", "date_issue", "document_id", "iban", "order_id", "payment_reference",\
        "payment_terms", "tax_detail_gross", "tax_detail_net", "tax_detail_rate", "tax_detail_tax", "vendor_address", "vendor_email",\
        "vendor_name", "vendor_order_id", "vendor_registration_id", "vendor_tax_id"]


class DoclieonlylirEntityVocab(DocTypeVocab):
    key_words = ["line_item_amount_gross", "line_item_amount_net", "line_item_code", "line_item_currency", "line_item_date", \
        "line_item_description", "line_item_discount_amount", "line_item_discount_rate",  "line_item_hts_number", "line_item_order_id", \
        "line_item_person_name", "line_item_position", "line_item_quantity", "line_item_tax", "line_item_tax_rate", \
        "line_item_unit_price_gross", "line_item_unit_price_net", "line_item_units_of_measure", "line_item_weight"]

class DoclieonlylirEntityVocabNewCe(DocTypeVocab):
    key_words = ["line_item_amount_gross", "line_item_amount_net", "line_item_code", "line_item_currency", "line_item_date", \
        "line_item_description", "line_item_discount_amount", "line_item_discount_rate",  "line_item_hts_number", "line_item_order_id", \
        "line_item_person_name", "line_item_position", "line_item_quantity", "line_item_tax", "line_item_tax_rate", \
        "line_item_unit_price_gross", "line_item_unit_price_net", "line_item_units_of_measure", "line_item_weight", "other"]

