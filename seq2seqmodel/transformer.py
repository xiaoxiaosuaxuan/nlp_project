#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

nn.Transformer

class transformer(nn.Module):
    
    def __init__(self, config):
        self.config = config
        self.d_model = 512
        self.device = 'cuda'
        self.decode_embed_size = 32
        self.nheads = 8 
        self.fc_fsize = 128 
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nheads, batch_first=True)
        self.encoder_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, max_len = 100)
        self.decoder_embed = nn.Embedding(self.decode_embed_size, config.num_tags, padding_idx=config.tag_pad_idx)
        self.fc_out = nn.Sequential(nn.Linear(self.d_model, self.fc_fsize), nn.Linear(self.fc_fsize, self.config.num_tags))
        self.loss =  nn.CrossEntropyLoss(ignore_index=config.tag_pad_idx)
        
    
    def forward(self, batch, mode='train'):
        if mode == 'train':
            batchsize = len(batch.lengths)
            src = self.encoder_embed(batch.input_ids.to(self.device))   # src = [N, L, src_embed_size]
            src = self.pos_encoding(src) 
            tgt_bos = torch.tensor([self.config.tag_bos_idx] * batchsize).rashape(batchsize, 1).to(self.device) 
            # tgt_bos = [N, 1]
            tgt_inputs = torch.cat((tgt_bos, batch.tag_ids[:, 0:-1].to(self.device)))  # tgt_inputs = [N, L] , after shift right
            tgt = self.decoder_embed(tgt_inputs)    # tgt = [N, L, tgt_embed_size]
            tgt = self.pos_encoding(tgt)   
            
            max_len = max(batch.lengths)
            padding_mask = [[False] * length + [True] * (max_len - length) for length in batch.lengths] 
            padding_mask = torch.tensor(padding_mask, device=self.device)   # padding_mask = [N, L]
            tgt_mask = torch.tril(torch.ones(max_len, max_len)).to(self.device)       
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('0'))      # tgt_mask = [L, L]
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
            batchsize = len(batch.lengths)
            src = self.encoder_embed(batch.input_ids.to(self.device))   # src = [N, L, src_embed_size]
            src = self.pos_encoding(src) 
            
            tgt = torch.tensor([self.config.tag_bos_idx] * batchsize).reshape(batchsize, 1).to(self.device)  
            # 一开始tgt只有bos，会随着decoder输出不断扩充
            max_len = max(batch.lengths)
            padding_mask = [[False] * length + [True] * (max_len - length) for length in batch.lengths] 
            padding_mask = torch.tensor(padding_mask, device=self.device)    # padding_mask = [N, L]
            
            memory = self.transformer.encoder(src, src_key_padding_mask=padding_mask)   # encoder 的输出 , [n, ]
            probs_logits = torch.empty((batchsize, max_len, self.config.num_tags))  # probs_logits = [N, L, num_tags] 用来存放接下来解码的输出 
            
            for i in range(max_len):
                cur_tgt = self.pos_encoding(self.decoder_embed(tgt))  # cur_tgt = [N, (i+1), tgt_embed_size]
                cur_padding_mask = padding_mask
            

            
            
        
        



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1, device='cuda'):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)
      self.device = device
      pe = torch.zeros((max_len, 1, d_model),
          dtype=torch.float32).to(self.device)
      for pos in range(0, max_len): 
          for i in range(0, d_model, 2):
              div_term = torch.exp(i * \
                -torch.log(10000.0) / d_model)
              pe[pos, 0, i] = torch.sin(pos * div_term)
              pe[pos, 0, i+1] = torch.cos(pos * div_term)

      self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)