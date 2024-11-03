import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, BertModel
import torch


class SpokenSQuADDataset(Dataset):
    def __init__(self, data_path, window_size=64, stride=32, question_length=32, phase='train'):
        self.window_size = window_size
        self.stride = stride
        self.question_length = question_length
        self.phase = phase
        self.data = self.load_data(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.phase == 'train':
            sample = self.preprocess_train(sample)
        elif self.phase == 'test':
            sample = self.preprocess_test(sample)
        return sample

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        all_data = data['data']
        samples = []
        for article in all_data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                qas = paragraph['qas']
                if len(qas) > 0:
                    for qa in qas:
                        question = qa['question']
                        answer = qa['answers'][0]['text']
                        answer_start = qa['answers'][0]['answer_start']
                        qid = qa['id']
                        samples.append({
                            'context': context,
                            'question': question,
                            'answer': answer,
                            'answer_start': answer_start,
                            'id': qid
                        })
        return samples

    def preprocess_train(self, sample):
        context, question, answer, answer_start, qid = sample['context'], sample['question'], sample['answer'], sample[
            'answer_start'], sample['id']
        # Tokenize context and question
        tokenized_context = self.tokenizer.encode(context, add_special_tokens=False, max_length=512, truncation=True)
        tokenized_question = self.tokenizer.encode(question, add_special_tokens=False, max_length=512, truncation=True)

        # Tokenize answer
        tokenized_answer = self.tokenizer.encode(answer, add_special_tokens=False, max_length=512, truncation=True)
        # Find start and end position of tokenized_answer in tokenized_context
        start_pos = -1
        end_pos = -1
        for i in range(len(tokenized_context) - len(tokenized_answer) + 1):
            if tokenized_context[i:i + len(tokenized_answer)] == tokenized_answer:
                start_pos = i
                end_pos = i + len(tokenized_answer)
                break
        if start_pos == -1:
            tokenized_before_answer = self.tokenizer.encode(context[:answer_start], add_special_tokens=False,
                                                            max_length=512, truncation=True)
            tokenized_by_answer = self.tokenizer.encode(context[:answer_start + len(answer)], add_special_tokens=False,
                                                        max_length=512, truncation=True)
            start_pos = len(tokenized_before_answer)
            end_pos = len(tokenized_by_answer)

        # Calculate the segment start and segment end
        segment_start = start_pos // self.stride * self.stride
        segment_end = segment_start + self.window_size
        # Pad the context if needed
        tokenized_context_segment = tokenized_context[segment_start:segment_end]
        context_segment_mask = [1] * len(tokenized_context_segment)
        question_mask = [1] * len(tokenized_question)
        start_pos -= segment_start
        end_pos -= segment_start
        if len(tokenized_context_segment) < self.window_size:
            tokenized_context_segment_ = tokenized_context_segment + [self.tokenizer.pad_token_id] * (
                        self.window_size - len(tokenized_context_segment))
            context_segment_mask_ = context_segment_mask + [0] * (self.window_size - len(tokenized_context_segment))
        else:
            tokenized_context_segment_ = tokenized_context_segment[:self.window_size]
            context_segment_mask_ = context_segment_mask[:self.window_size]
        # Pad the question if needed
        if len(tokenized_question) < self.question_length:
            tokenized_question_ = tokenized_question + [self.tokenizer.pad_token_id] * (
                        self.question_length - len(tokenized_question))
            question_mask_ = question_mask + [0] * (self.question_length - len(tokenized_question))
        else:
            tokenized_question_ = tokenized_question[:self.question_length]
            question_mask_ = question_mask[:self.question_length]
        input_ids = tokenized_question_ + [self.tokenizer.sep_token_id] + tokenized_context_segment_
        input_mask = question_mask_ + [1] + context_segment_mask_
        # print(self.tokenizer.decode(input_ids))
        # print(answer)
        # print(self.tokenizer.decode(input_ids[start_pos+self.question_length+1:end_pos+self.question_length+1]))
        # print(start_pos, end_pos)
        return {
            'input_ids': torch.tensor(input_ids),
            'input_mask': torch.tensor(input_mask),
            'start_pos': torch.tensor(start_pos),
            'end_pos': torch.tensor(end_pos),
            'id': qid
        }

    def preprocess_test(self, sample):
        # use a sliding window approach to process the context
        # the returned sample should be a list of dictionaries
        context, question, answer, answer_start, qid = sample['context'], sample['question'], sample['answer'], sample[
            'answer_start'], sample['id']
        # Tokenize context and question
        tokenized_context = self.tokenizer.encode(context, add_special_tokens=False, max_length=512, truncation=True)
        tokenized_question = self.tokenizer.encode(question, add_special_tokens=False, max_length=512, truncation=True)
        question_mask = [1] * len(tokenized_question)
        if len(tokenized_question) < self.question_length:
            tokenized_question_ = tokenized_question + [self.tokenizer.pad_token_id] * (
                        self.question_length - len(tokenized_question))
            question_mask_ = question_mask + [0] * (self.question_length - len(tokenized_question))
        else:
            tokenized_question_ = tokenized_question[:self.question_length]
            question_mask_ = question_mask[:self.question_length]

        # Tokenize answer
        tokenized_answer = self.tokenizer.encode(answer, add_special_tokens=False, max_length=512, truncation=True)
        # Find start and end position of tokenized_answer in tokenized_context
        start_pos = -1
        end_pos = -1
        for i in range(len(tokenized_context) - len(tokenized_answer) + 1):
            if tokenized_context[i:i + len(tokenized_answer)] == tokenized_answer:
                start_pos = i
                end_pos = i + len(tokenized_answer)
                break
        if start_pos == -1:
            tokenized_before_answer = self.tokenizer.encode(context[:answer_start], add_special_tokens=False,
                                                            max_length=512, truncation=True)
            tokenized_by_answer = self.tokenizer.encode(context[:answer_start + len(answer)], add_special_tokens=False,
                                                        max_length=512, truncation=True)
            start_pos = len(tokenized_before_answer)
            end_pos = len(tokenized_by_answer)

        all_input_ids = []
        all_input_mask = []
        for i in range(0, len(tokenized_context) - self.window_size + 1, self.stride):
            segment_start = i
            segment_end = i + self.window_size
            tokenized_context_segment = tokenized_context[segment_start:segment_end]
            context_segment_mask = [1] * len(tokenized_context_segment)
            start_pos_ = start_pos - segment_start
            end_pos_ = end_pos - segment_start
            if len(tokenized_context_segment) < self.window_size:
                tokenized_context_segment_ = tokenized_context_segment + [self.tokenizer.pad_token_id] * (
                            self.window_size - len(tokenized_context_segment))
                context_segment_mask_ = context_segment_mask + [0] * (self.window_size - len(tokenized_context_segment))
            else:
                tokenized_context_segment_ = tokenized_context_segment[:self.window_size]
                context_segment_mask_ = context_segment_mask[:self.window_size]

            input_ids = tokenized_question_ + [self.tokenizer.sep_token_id] + tokenized_context_segment_
            input_mask = question_mask_ + [1] + context_segment_mask_
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
        return {
            'input_ids': torch.tensor(all_input_ids),
            'input_mask': torch.tensor(all_input_mask),
            'start_pos': torch.tensor(start_pos),
            'end_pos': torch.tensor(end_pos),
            'answer': answer,
            'id': qid
        }


if __name__ == "__main__":
    data_path = './Spoken-SQuAD/spoken_train-v1.1.json'
    dataset = SpokenSQuADDataset(data_path)
    print(len(dataset))
    for i in range(100):
        sample = dataset[i]