#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import math

nn.Transformer

class transformer(nn.Module):
    
    def __init__(self, config):
        super(transformer, self).__init__()
        self.config = config
        self.d_model = config.embed_size
        self.device = 'cuda'
        self.nheads = 8 
        self.fc_fsize = 256  
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nheads, batch_first=True)
        self.encoder_embed = nn.Embedding(config.vocab_size, self.d_model, padding_idx=config.pad_idx)
        self.decoder_embed = nn.Embedding(config.num_tags, self.d_model, padding_idx=config.tag_pad_idx)
        self.fc_out = nn.Sequential(nn.Linear(self.d_model, self.fc_fsize), nn.Linear(self.fc_fsize, self.config.num_tags))
        self.loss =  nn.CrossEntropyLoss(ignore_index=config.tag_pad_idx)
        
    
    def forward(self, batch, mode='train'):
        batchsize = len(batch.lengths)
        max_len = max(batch.lengths)
        src = self.encoder_embed(batch.input_ids.to(self.device))   # src = [N, L, d_model]
        pos_encoding = PositionalEncoding(d_model=self.d_model, max_len=max_len)
        src = pos_encoding(src) 
        
        if mode == 'train':
            tgt_bos = torch.tensor([self.config.tag_bos_idx] * batchsize).reshape(batchsize, 1).to(self.device) 
            # tgt_bos = [N, 1]
            tgt_inputs = torch.cat((tgt_bos, batch.tag_ids[:, :-1].to(self.device)), dim=-1)  # tgt_inputs = [N, L] , after shift right
            tgt = self.decoder_embed(tgt_inputs)    # tgt = [N, L, d_model]
            tgt = pos_encoding(tgt)   
            
            padding_mask = [[False] * length + [True] * (max_len - length) for length in batch.lengths] 
            padding_mask = torch.tensor(padding_mask, device=self.device)   # padding_mask = [N, L]
            tgt_mask = torch.tril(torch.ones(max_len, max_len)).to(self.device)       
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask==1, float('0'))     # tgt_mask = [L, L]
            tgt_mask = tgt_mask.repeat(self.nheads*batchsize, 1, 1)     # tgt_mask = [nheads * N, L, L]
            
            outputs = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=padding_mask, 
                                       tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask) 
            # outputs  = [N, L, d_model]
            probs_logits = self.fc_out(outputs)       # probs = [N, L, num_tags], not softmaxed 
            tag_mask = batch.tag_mask.to(self.device)
            probs_logits += (1 - tag_mask).unsqueeze(-1).repeat(1, 1, self.config.num_tags) * -1e32   # 避免超过length的维度对softmax 产生干扰
            probs = torch.softmax(probs_logits, dim=-1)  # probs = [N, L, num_tags]  , after softmax, get the real probablity
            tag_ids = batch.tag_ids.to(self.device)  
            loss = self.loss(probs_logits.view(-1, probs_logits.shape[-1]), tag_ids.view(-1)) 
            return probs, loss 
          
        elif mode == 'test':
            tgt = torch.tensor([self.config.tag_bos_idx] * batchsize).reshape(batchsize, 1).to(self.device)  
            # 一开始tgt只有bos，会随着decoder输出不断扩充
            padding_mask = [[False] * length + [True] * (max_len - length) for length in batch.lengths] 
            padding_mask = torch.tensor(padding_mask, device=self.device)    # padding_mask = [N, L]
            
            memory = self.transformer.encoder(src, src_key_padding_mask=padding_mask)   # encoder 的输出 , [N,L,d_model]
            probs_logits = torch.empty((batchsize, max_len, self.config.num_tags))  # probs_logits = [N, L, num_tags] 用来存放接下来解码的输出 
            
            for i in range(max_len):
                cur_tgt = pos_encoding(self.decoder_embed(tgt))  # cur_tgt = [N, (i+1), d_model]
                cur_padding_mask = padding_mask[:, :i+1]
                output = self.transformer.decoder(tgt=cur_tgt, memory=memory, tgt_key_padding_mask=cur_padding_mask,
                                                  memory_key_padding_mask=cur_padding_mask)  # output = [N, (i+1), d_model]
                prob =  self.fc_out(output[:, -1, :].squeeze())    # prob = [N, num_tags]
                probs_logits[:, i, :] = prob    
                predict_tgt = torch.argmax(prob, dim=-1).reshape(batchsize, 1)   # predict_tgt = [N, 1]
                tgt = torch.cat((tgt, predict_tgt), dim=-1)          # 用预测的标签扩充tgt
                
            probs_logits += (1 - tag_mask).unsqueeze(-1).repeat(1, 1, self.config.num_tags) * -1e32   # 避免超过length的维度对softmax 产生干扰
            probs = torch.softmax(probs_logits, dim=-1)  # probs = [batch_size * Length * num_tags]  , after softmax, get the real probablity
            return probs
                
    def decode(self, label_vocab, batch, mode='train'):       #label_vocab: id2tags
        batch_size = len(batch)
        labels = batch.labels
        if mode == 'train':
            prob, loss = self.forward(batch, mode)
        elif mode == 'test':
            prob = self.forward(batch, mode)
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
        if mode == 'train':
            return predictions, labels, loss.cpu().item()
        elif mode == 'test':
            return predictions, labels

            
            
        
        



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1, device='cuda'):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)
      self.device = device
      pe = torch.zeros((max_len,  d_model),
          dtype=torch.float32).to(self.device)
      for pos in range(0, max_len): 
          for i in range(0, d_model, 2):
              div_term = math.exp(i * \
                -math.log(10000.0) / d_model)
              pe[pos, i] = math.sin(pos * div_term)
              pe[pos, i+1] = math.cos(pos * div_term)

      self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[1], :]
        return self.dropout(x)