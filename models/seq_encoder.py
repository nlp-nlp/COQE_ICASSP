import torch.nn as nn
from transformers import BertModel


class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_directory)
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, pooler_output = out.last_hidden_state, out.pooler_output
        return last_hidden_state, pooler_output