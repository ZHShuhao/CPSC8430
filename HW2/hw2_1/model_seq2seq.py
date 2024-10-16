import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, frame_dim=4096, hidden=256, dropout=0.2, v_step=80, c_step=50, bos_idx=0):
        super(Seq2Seq, self).__init__()
        self.frame_dim = frame_dim
        self.hidden = hidden
        self.v_step = v_step
        self.c_step = c_step
        self.bos_idx = bos_idx
        self.vocab_size = vocab_size

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(frame_dim, hidden)
        self.linear2 = nn.Linear(hidden, vocab_size)

        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(2 * hidden, hidden, batch_first=True, dropout=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden)

    def forward(self, video, captions=None):
        batch_size = video.shape[0]
        video = video.contiguous().view(-1, self.frame_dim)
        video = self.drop(video)
        video = self.linear1(video)
        video = video.view(-1, self.v_step, self.hidden)

        padding = torch.zeros([batch_size, self.c_step - 1, self.hidden]).cuda()
        video = torch.cat((video, padding), 1)
        vid_out, state_vid = self.lstm1(video)

        padding = torch.zeros([batch_size, self.v_step, self.hidden]).cuda()
        cap_input = torch.cat((padding, vid_out[:, 0:self.v_step, :]), 2)
        cap_out, state_cap = self.lstm2(cap_input)

        bos_id = self.bos_idx * torch.ones(batch_size, dtype=torch.long).cuda()
        cap_input = self.embedding(bos_id)
        cap_input = torch.cat((cap_input, vid_out[:, self.v_step, :]), 1)
        cap_input = cap_input.view(batch_size, 1, 2 * self.hidden)

        cap_out, state_cap = self.lstm2(cap_input, state_cap)
        cap_out = cap_out.contiguous().view(-1, self.hidden)
        cap_out = self.drop(cap_out)
        cap_out = self.linear2(cap_out)

        cap_prob = [cap_out]
        cap_out = torch.argmax(cap_out, 1)
        caption = [cap_out]

        for i in range(self.c_step - 2):
            cap_input = self.embedding(cap_out)
            cap_input = torch.cat((cap_input, vid_out[:, self.v_step + 1 + i, :]), 1)
            cap_input = cap_input.view(batch_size, 1, 2 * self.hidden)

            cap_out, state_cap = self.lstm2(cap_input, state_cap)
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            cap_prob.append(cap_out)
            cap_out = torch.argmax(cap_out, 1)
            caption.append(cap_out)

        caption = torch.stack(caption, 1)
        cap_prob = torch.stack(cap_prob, 1)
        return caption, cap_prob


if __name__ == '__main__':
    num_vfeatures = 4096
    num_words = 5978
    length_vfeatures = 80
    length_sentence = 50
    bs = 8

    bos_idx = 0  # Assuming <BOS> token index is 0 for simplicity
    model = Seq2Seq(num_words, frame_dim=num_vfeatures, hidden=256, dropout=0.2, v_step=length_vfeatures,
                    c_step=length_sentence, bos_idx=bos_idx)
    vfeatures = torch.randn(bs, length_vfeatures, num_vfeatures).cuda()

    model = model.cuda()
    model.train()
    caption, cap_prob = model(vfeatures)
    print("Caption Shape:", caption.shape)
    print("Cap Prob Shape:", cap_prob.shape)

    model.eval()
    caption, cap_prob = model(vfeatures)
    print("Caption Shape (eval):", caption.shape)
    print("Cap Prob Shape (eval):", cap_prob.shape)