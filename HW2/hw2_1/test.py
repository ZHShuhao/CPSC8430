import os
from model_seq2seq import Seq2Seq
import torch
import json
from argparse import ArgumentParser
import numpy as np
import re

import os
print(os.path.isfile('/scratch/shuhaoz/DLHW/HW2/MLDS_hw2_1_data/testing_data/id.txt'))


class Text2Embedding():
    def __init__(self, word_dict, max_length=50):
        try:
            self.word2id = json.load(open(word_dict, 'r'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Word dictionary file '{word_dict}' not found.")
        self.max_length = max_length
        self.num_words = len(self.word2id)
        self.num_captions = 20
        self.id2word = {v: k for k, v in self.word2id.items()}

    def _parse_sentence(self, sentence):
        pattern = r'\w+|\S'
        tokens = re.findall(pattern, sentence.lower())
        tokens.insert(0, '<BOS>')
        tokens.append('<EOS>')
        return tokens

    def __call__(self, captions):
        embeddings = []
        for caption in captions:
            embedding = []
            tokens = self._parse_sentence(caption)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens += ['<PAD>'] * (self.max_length - len(tokens))

            for token in tokens:
                one_hot = self.word2id.get(token, self.word2id['<UNK>'])
                embedding.append(one_hot)
            embeddings.append(embedding)
        if len(embeddings) < self.num_captions:
            num_replicas = self.num_captions - len(embeddings)
            embeddings += [embeddings[i % len(embeddings)] for i in range(num_replicas)]
        embeddings = embeddings[:self.num_captions]
        return torch.LongTensor(embeddings)

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/shuhaoz/DLHW/HW2/MLDS_hw2_1_data/testing_data')
    parser.add_argument('--output', type=str, default='/scratch/shuhaoz/DLHW/HW2/output.txt')
    return parser.parse_args()


def build_model(checkpoint):
    num_vfeatures = 4096
    num_words = 5978
    length_vfeatures = 80
    length_sentence = 50
    embed = Text2Embedding('word_dict.json', max_length=length_sentence)
    bos_idx = embed.word2id['<BOS>']
    id2word = embed.id2word
    model = Seq2Seq(num_words, frame_dim=num_vfeatures, hidden=256, dropout=0.2, v_step=length_vfeatures,
                    c_step=length_sentence, bos_idx=bos_idx)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    model = model.cuda()
    return model, id2word


def predict(model, testing_dir, output_file, id2word):
    try:
        test_ids = os.path.join(testing_dir, 'id.txt')
        test_ids = open(test_ids, 'r').read().splitlines()
    except:
        raise FileNotFoundError('id.txt not found in testing_dir')
    with open(output_file, 'w') as file:
        for id in test_ids:
            vfeat = torch.from_numpy(np.load(os.path.join(testing_dir, 'feat', f'{id}.npy'))).float()
            vfeat = vfeat.unsqueeze(0).cuda()
            output, pred = model(vfeat)
            output = output.squeeze(0).cpu().detach().numpy()
            sentence = []
            for word_id in output:
                word = id2word[word_id]
                if word == '<EOS>' or word == '<PAD>' or word == '.':
                    break
                sentence.append(word)
            sentence = ' '.join(sentence)
            file.write(f'{id},{sentence}\n')


if __name__ == '__main__':
    args = arg_parser()
    checkpoint = 'final_model.pt'
    google_drive_link = ''
    if not os.path.exists(checkpoint):
        print(f'{checkpoint} not found. Downloading from Google Drive...')
        os.system(f'wget --no-check-certificate "{google_drive_link}" -O {checkpoint}')
        print('Download complete.')
    model, id2word = build_model(checkpoint)
    predict(model, args.data_dir, args.output, id2word)
    print(f'Predictions saved to {args.output}.')
