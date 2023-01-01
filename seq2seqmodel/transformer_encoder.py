#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

nn.Transformer

class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.d_model = config.embed_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.output_layer = TaggingFNNDecoder(self.d_model, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        batchsize = len(batch.lengths)
        src = self.encoder_embed(batch.input_ids.to(self.device))   # src = [N, L, src_embed_size]
        src = self.pos_encoding(src) 
        
        # 一开始tgt只有bos，会随着decoder输出不断扩充
        max_len = max(batch.lengths)
        padding_mask = [[False] * length + [True] * (max_len - length) for length in batch.lengths] 
        padding_mask = torch.tensor(padding_mask, device=self.device)    # padding_mask = [N, L]
        
        memory = self.encoder(src, src_key_padding_mask=padding_mask)   # encoder 的输出 , [N,L,d_model]
        
        tag_ids = batch.tag_ids        #N*L
        tag_mask = batch.tag_mask       #N*L
        
        tag_output = self.output_layer(memory, tag_mask, tag_ids)

        return tag_output

    
    def decode(self, label_vocab, batch):       #label_vocab: id2tags
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()       #input_size: the size of hidden feature of RNN in each time step
        self.num_tags = num_tags                   #number of classes of classification, (num of tags)
        self.output_layer = nn.Linear(input_size, num_tags)         
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)    #N*L*C     (batch_size * length of sentence * classes of tags )
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32 
        
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

