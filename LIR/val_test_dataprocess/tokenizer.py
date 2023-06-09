from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
english_tokenizer = AutoTokenizer.from_pretrained('https://huggingface.co/sentence-transformers/roberta-base-nli-stsb-mean-tokens')
english_bert = AutoModel.from_pretrained('https://huggingface.co/sentence-transformers/roberta-base-nli-stsb-mean-tokens').cuda().eval()
