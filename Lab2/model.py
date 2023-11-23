import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from itertools import zip_longest
from utils import indexed


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, out_size: int, corpus: dict, emb_size: int = 512, hidden_size: int = 512):
        """
        args:
            vocab_size: size of corpus
            out_size: size of tag-set
            emb_size: dimension of embedding
            hidden_size: dimension of hidden layer
        """
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.out_size = out_size
        self.corpus = corpus
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2*hidden_size, out_size)
        self.transition = nn.Parameter(torch.rand(out_size, out_size)*1/out_size) # CRF transition matrix [out_size, out_size]

    def forward(self, sentences, lengths):
        emb = self.embedding(sentences)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=True)
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)  # [batch_size, max_len, out_size]
        # CRF scores[batch_size, max_len, out_size, out_size], every wordcorresponding to a [out_size, out_size] matrix
        # [i,j] means the score: tag of last word is i and tag of current word is j
        crf_scores = scores.unsqueeze(2).expand(-1, -1, self.out_size, -1) + self.transition.unsqueeze(0)
        return crf_scores

    def test(self, sentences, lengths, tag2id):
        """ 使用维特比算法Decode """
        start_id = tag2id['<start>']
        stop_id = tag2id['<stop>']
        pad = tag2id['<pad>']
        crf_scores = self.forward(sentences, lengths)
        device = crf_scores.device
        B, L, T, _ = crf_scores.size() # B:batch_size, L:max_len, T:self.out_size
        # viterbi[i, j, k]: maximum score of word j in sentence i is labeled tag k
        viterbi = torch.zeros(B, L, T).to(device) 
        # backpointer[i, j, k]: id of last tag. current is word j in sentence i is labeled tag k
        backpointer = (torch.zeros(B, L, T).long() * stop_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        for step in range(L):
            valid_size_t = (lengths > step).sum().item()
            if step == 0: # tag before first word is <start>
                viterbi[:valid_size_t, step,:] = crf_scores[: valid_size_t, step, start_id, :]
                backpointer[: valid_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:valid_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:valid_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:valid_size_t, step, :] = max_scores
                backpointer[:valid_size_t, step, :] = prev_tags
        # find path
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []
        tags_t = None
        for step in range(L-1, 0, -1):
            valid_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(valid_size_t).long() * (step * self.out_size)
                index = index.to(device)
                index += stop_id
            else:
                prev_valid_size_t = len(tags_t)
                new_in_batch = torch.LongTensor([stop_id] * (valid_size_t - prev_valid_size_t)).to(device)
                offset = torch.cat([tags_t, new_in_batch], dim=0)
                index = torch.ones(valid_size_t).long() * (step * self.out_size)
                index = index.to(device)
                index += offset.long()
            tags_t = backpointer[:valid_size_t].gather(dim=1, index=index.unsqueeze(1).long())
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()
        return tagids
    
    def loss(self, crf_scores, targets, tag2id):
        """ calculate loss for BiLSTM-CRF (https://arxiv.org/pdf/1603.01360.pdf) """
        pad_id = tag2id['<pad>']
        start_id = tag2id['<start>']
        end_id = tag2id['<stop>']
        device = crf_scores.device
        # targets:[B, L] crf_scores:[B, L, T, T]
        batch_size, max_len = targets.size()
        target_size = len(tag2id)
        # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
        mask = (targets != pad_id)
        lengths = mask.sum(dim=1)
        targets = indexed(targets, target_size, start_id)
        # All path scores. scores_sum_t[i, j]: word t of sentence i labeled label j, sum of scores before word t
        scores_sum_t = torch.zeros(batch_size, target_size).to(device)
        for t in range(max_len):
            valid_size_t = (lengths > t).sum().item()
            if t == 0:
                scores_sum_t[:valid_size_t] = crf_scores[:valid_size_t, t, start_id, :]
            else:
                # transition had been added to crf_score in forward process
                scores_sum_t[:valid_size_t] = torch.logsumexp(
                    crf_scores[:valid_size_t, t, :, :] + scores_sum_t[:valid_size_t].unsqueeze(2),
                    dim = 1
                )
        all_path_scores = scores_sum_t[:, end_id].sum()
        # Golden scores
        targets = targets.masked_select(mask)
        flatten_scores = crf_scores.masked_select(mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores))
        flatten_scores = flatten_scores.view(-1, target_size*target_size).contiguous()
        golden_scores = flatten_scores.gather(dim=1, index=targets.unsqueeze(1)).sum()
        loss = (all_path_scores - golden_scores) / batch_size
        return loss