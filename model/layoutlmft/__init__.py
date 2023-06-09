from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import auto_class_factory

from .models.layoutlmv2 import (
    LayoutLMv2Config,
    LayoutLMv2ForRelationExtraction,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LayoutLMv2TokenizerFast,
)
from .models.layoutxlm import (
    LayoutXLMConfig,
    LayoutXLMForRelationExtraction,
    LayoutXLMForTokenClassification,
    LayoutXLMTokenizer,
    LayoutXLMTokenizerFast,
)
from .models.graphdoc import (
    GraphDocConfig, 
    GraphDocForTokenClassification
)

CONFIG_MAPPING.update([("layoutlmv2", LayoutLMv2Config), ("layoutxlm", LayoutXLMConfig), ("graphdoc", GraphDocConfig)])

MODEL_NAMES_MAPPING.update([("layoutlmv2", "LayoutLMv2"), ("layoutxlm", "LayoutXLM"), ("graphdoc", "GraphDoc")])

TOKENIZER_MAPPING.update(
    [
        (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)),
        (LayoutXLMConfig, (LayoutXLMTokenizer, LayoutXLMTokenizerFast))
    ]
)

SLOW_TO_FAST_CONVERTERS.update(
    {
        "LayoutLMv2Tokenizer": BertConverter, 
        "LayoutXLMConverter": XLMRobertaConverter
    }
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [
        (LayoutLMv2Config, LayoutLMv2ForTokenClassification), 
        (LayoutXLMConfig, LayoutXLMForTokenClassification),
        (GraphDocConfig, GraphDocForTokenClassification)
    ]
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict([
    (LayoutLMv2Config, LayoutLMv2ForRelationExtraction), 
    (LayoutXLMConfig, LayoutXLMForRelationExtraction)
])

AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

AutoModelForRelationExtraction = auto_class_factory(
    "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
)
