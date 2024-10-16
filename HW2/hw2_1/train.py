import torch
from model_seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
from torch.utils.data import Dataset
import json
import numpy as np
import re

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

class MyDataset(Dataset):
    def __init__(self, data_dir, split='training', pre_extracted=True, caption_transform=None):
        self.pre_extracted = pre_extracted
        self.split = split
        self.caption_transform = caption_transform
        if pre_extracted:
            self.data_folder = os.path.join(data_dir, split + '_data', 'feat')
        else:
            self.data_folder = os.path.join(data_dir, split + '_data', 'video')
        annotation_file = os.path.join(data_dir, split + '_label.json')
        try:
            self.data = json.load(open(annotation_file, 'r'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file '{annotation_file}' not found.")

    def __len__(self):
        return len(self.data)

    def load_video(self, video_path):
        # Implement video loading logic here if needed
        raise NotImplementedError("Video loading not implemented. Ensure `pre_extracted=True` or implement `load_video` method.")

    def __getitem__(self, idx):
        item = self.data[idx]
        id, caption = item['id'], item['caption']
        if self.pre_extracted:
            vfeat = torch.from_numpy(np.load(os.path.join(self.data_folder, id + '.npy'))).float()
        else:
            vfeat = self.load_video(os.path.join(self.data_folder, id))
        if self.caption_transform:
            caption = self.caption_transform(caption)
        return vfeat, caption


def plot_loss(losses_train):
    # Plot loss curve
    fig = plt.figure()
    plt.plot(losses_train, label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('loss_curve.png')
    plt.close()

def main():
    batch_size = 128
    num_workers = 8
    caption_length = 50
    video_length = 80
    num_vfeatures = 4096

    data_dir = '/scratch/shuhaoz/DLHW/HW2/MLDS_hw2_1_data'
    embed = Text2Embedding('word_dict.json', max_length=caption_length)

    dataset = MyDataset(data_dir, 'training', pre_extracted=True, caption_transform=embed)
    print(f"Dataset size: {len(dataset)}")
    for i in range(10):
        vfeat, caption = dataset[i]
        print(f"Video Feature Shape: {vfeat.shape}, Caption Shape: {caption.shape}")

    train_dataset = MyDataset(data_dir, 'training', pre_extracted=True, caption_transform=embed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_dir = '/scratch/shuhaoz/DL HW/HW2/MLDS_hw2_1_data'
    embed = Text2Embedding('word_dict.json', max_length=caption_length)


    num_words = embed.num_words
    bos_idx = embed.word2id['<BOS>']
    eos_idx = embed.word2id['<EOS>']
    pad_idx = embed.word2id['<PAD>']
    model = Seq2Seq(num_words, frame_dim=num_vfeatures, hidden=256, dropout=0.2, v_step=video_length,
                    c_step=caption_length, bos_idx=bos_idx)
    try:
        model.load_state_dict(torch.load('final_model.pt'))
        print('Model loaded from final_model.pt')
    except:
        pass
    model = model.cuda()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 500
    save_interval = 10

    losses_train = []
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        epoch_loss_train = 0
        for i, (vfeat, caption) in progress_bar:
            vfeat = vfeat.cuda()
            caption = caption.cuda()
            num_captions = caption.size(1)
            # num_captions = 1
            iter_loss = 0
            for j in range(num_captions):
                cur_caption = caption[:, j, :].cuda()
                output, prob = model(vfeat)

                prob = prob.view(-1, prob.shape[-1])
                target = cur_caption[:, 1:].contiguous().view(-1)
                loss = criterion(prob, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_loss += loss.item()
            iter_loss = iter_loss / num_captions
            progress_bar.set_description(f'Epoch {epoch} batch {i} loss: {iter_loss}')
            epoch_loss_train += iter_loss
        epoch_loss_train /= len(train_loader)
        print(f'Epoch {epoch} loss: {epoch_loss_train}')
        losses_train.append(epoch_loss_train)

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f'./final_model.pt')
            plot_loss(losses_train)

    torch.save(model.state_dict(), f'./final_model.pt')
    plot_loss(losses_train)



if __name__ == '__main__':
    main()
