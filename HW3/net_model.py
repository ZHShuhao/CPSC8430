import torch
from transformers import BertModel

import torch.nn as nn

class QANet(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=768, window_size=64, question_length=32, dropout=0.5):
        super(QANet, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        self.question_length = question_length
        self.start_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.end_classifier = nn.Sequential(
            nn.Linear(hidden_size+1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, input_masks):
        outputs = self.bert(input_ids, attention_mask=input_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)[:,self.question_length+1:,:]
        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(torch.cat([sequence_output, start_logits], dim=-1))
        return start_logits.squeeze(-1), end_logits.squeeze(-1)