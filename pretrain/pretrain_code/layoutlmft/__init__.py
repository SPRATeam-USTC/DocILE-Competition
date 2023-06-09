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
from .models.layoutclmV30 import (
    LayoutCLMv30Config,
    LayoutCLMv30ForRelationExtraction,
    LayoutCLMv30ForTokenClassification,
    LayoutCLMv30Tokenizer,
    LayoutCLMv30TokenizerFast,
)

CONFIG_MAPPING.update([("layoutlmv2", LayoutLMv2Config), ("layoutclmV30", LayoutCLMv30Config)])

MODEL_NAMES_MAPPING.update([("layoutlmv2", "LayoutLMv2"), ("layoutclmV30", "LayoutCLMv30")])

TOKENIZER_MAPPING.update(
    [
        (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)),
        (LayoutCLMv30Config, (LayoutCLMv30Tokenizer, LayoutCLMv30TokenizerFast))
    ]
)

SLOW_TO_FAST_CONVERTERS.update(
    {
        "LayoutLMv2Tokenizer": BertConverter,
        "LayoutCLMv30Converter": BertConverter
    }
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [
        (LayoutLMv2Config, LayoutLMv2ForTokenClassification), 
        (LayoutCLMv30Config, LayoutCLMv30ForTokenClassification)
    ]
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict([
    (LayoutLMv2Config, LayoutLMv2ForRelationExtraction),
    (LayoutCLMv30Config, LayoutCLMv30ForRelationExtraction)
])

AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

AutoModelForRelationExtraction = auto_class_factory(
    "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
)
