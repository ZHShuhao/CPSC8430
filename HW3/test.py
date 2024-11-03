import os
import torch
from net_model import QANet
from dataset import SpokenSQuADDataset
from tqdm import tqdm
from transformers import AutoTokenizer
import collections


def postprocess(start_logits, end_logits, input_ids, tokenizer):
    window_size = 64
    question_length = 32
    input_ids = input_ids[:, question_length + 1:]
    start_logits = torch.softmax(start_logits, dim=-1)
    end_logits = torch.softmax(end_logits, dim=-1)

    batch_size = start_logits.shape[0]
    best_answer_prob = 0
    best_answer_pos = (0, 0)
    best_answer_batch_idx = 0

    for i in range(batch_size):
        for j in range(window_size):
            for k in range(j, window_size):
                prob = start_logits[i][j] + end_logits[i][k]
                if prob > best_answer_prob:
                    best_answer_prob = prob
                    best_answer_pos = (j, k)
                    best_answer_batch_idx = i
    answer_start, answer_end = best_answer_pos
    answer = tokenizer.decode(input_ids[best_answer_batch_idx][answer_start:answer_end], skip_special_tokens=True)
    return answer


def eval_f1(gt_answers, pred_answers):
    # calculate the question answer F1 score between gt_answers and pred_answers
    f1_scores = []
    for gt_answer, pred_answer in zip(gt_answers, pred_answers):
        common = collections.Counter(gt_answer) & collections.Counter(pred_answer)
        num_common = sum(common.values())
        if num_common == 0:
            f1_scores.append(0)
            continue
        precision = num_common / len(pred_answer)
        recall = num_common / len(gt_answer)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    return avg_f1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = QANet()
    ckpt_path = 'qanet_final.pt'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f'Load model from {ckpt_path}')
    model.to(device)

    test_data_path = './Spoken-SQuAD/spoken_test-v1.1.json'
    test_dataset = SpokenSQuADDataset(test_data_path, phase='test')
    gt_answers = []
    pred_answers = []

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        input_ids, input_mask, answer, qid = sample['input_ids'], sample['input_mask'], sample['answer'], sample['id']
        # print(input_ids.shape, input_mask.shape)
        if len(input_ids) == 0:
            continue
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        start_logits, end_logits = model(input_ids, input_mask)
        final_answer = postprocess(start_logits, end_logits, input_ids, tokenizer)
        print(f'{i}/{len(test_dataset)} GT answer: {answer}; \t \t Predicted answer: {final_answer}')

        gt_answers.append(answer)
        pred_answers.append(final_answer)

    f1_score = eval_f1(gt_answers, pred_answers)
    print(f'F1 score: {f1_score}')


if __name__ == "__main__":
    main()