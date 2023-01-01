#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config 
        self.device = "cuda" if config.device >= 0 else "cpu"
        self.embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = nn.LSTM(config.embed_size, config.hidden_size//2 , num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=config.dropout)
        
    def forward(self, inputs, lengths):
        embeds = self.dropout(self.embed(inputs))
        pack_embeds = rnn_utils.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=True)
        hiddens, h_t_c_t = self.rnn(pack_embeds)
        hiddens, _ = rnn_utils.pad_packed_sequence(hiddens, batch_first=True)
        return hiddens, h_t_c_t 

class AttnLayer(nn.Module):
    
    def __init__(self, input_size):
        super(AttnLayer, self).__init__()
        self.fsize = 128
        self.W_e = nn.Linear(input_size, self.fsize)
        self.W_d = nn.Linear(input_size, self.fsize, bias=False)
        self.v = nn.Linear(self.fsize, 1)
        
    def forward(self, e_hiddens, d_hiddens, length_mask):
        e = self.W_e(e_hiddens)     # e = [batch_size * Length * fsize]
        d = self.W_d(d_hiddens)     # d = [batch_size * fsize]
        d_hiddens = d.unsqueeze(dim=1) + e   # d_hiddens = [batch_size * Length * fsize]
        d_hiddens = torch.tanh(d_hiddens) 
        d_hiddens = self.v(d_hiddens).squeeze()      # d_hiddens = [batch_size * Length]  ,length_mask = [batch_size * Length]
        d_hiddens = d_hiddens + (1 - length_mask) * 1e-32  # 避免超过句子长度的维度在softmax时产生干扰
        d_hiddens = torch.softmax(d_hiddens, dim=-1).unsqueeze(dim=1)  # d_hiddens = [batch_size * 1 * Length] , e_hiddens = [batch_size * Length * hidden_size]
        d_hiddens = torch.bmm(d_hiddens, e_hiddens).squeeze()   # d_hiddens = [batch_size * hidden_size]  , the weighted added h_i
        return d_hiddens
        
        

class Decoder(nn.Module):
    
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config 
        self.device = "cuda" if config.device >= 0 else "cpu"
        self.num_tags = config.num_tags
        self.embed_size = 16
        self.embed = nn.Linear(config.num_tags, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, config.hidden_size, num_layers=config.num_layer, bidirectional=False, batch_first=True)
        self.attn = AttnLayer(config.hidden_size)
        self.fc_out_fsize = 128
        self.fc_out = nn.Sequential(nn.Linear(config.hidden_size * 2, self.fc_out_fsize), nn.Linear(self.fc_out_fsize, self.num_tags))
        
    def forward(self, tag_inputs, h_t_c_t, e_hiddens, length_mask):      # tag_inputs = [batch_size]
        onehot_inputs = nn.functional.one_hot(tag_inputs, num_classes = self.num_tags).to(self.device)  # onehot = [batch_size * num_tags]
        embeds = self.embed(onehot_inputs.to(torch.float32))    # embeds = [batch_size * embed_size]
        embeds = embeds.unsqueeze(dim=1)      # embeds = [batch_size * 1 * embed_size]
        d_hiddens, h_t_c_t = self.rnn(embeds, h_t_c_t)
        d_hiddens = d_hiddens.squeeze()            # d_hiddens = [batch_size * hidden_size]    e_hiddens = [batch_size * Length * hidden_size]
        h_cxt = self.attn(e_hiddens, d_hiddens, length_mask)    # h_cxt = [batch_size * hidden_size]
        probs = self.fc_out(torch.cat((h_cxt, d_hiddens), dim=-1))  # probs_ [batch_size * num_tags], not softmaxed
        return probs, h_t_c_t
    
class AttnSeq2seq(nn.Module):
    
    def __init__(self, config):
        super(AttnSeq2seq, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.device = "cuda" if config.device >= 0 else "cpu"
        self.bos_id = config.tag_bos_idx
        self.loss = nn.CrossEntropyLoss(ignore_index=config.tag_pad_idx)
    
  
        
    def forward(self, batch, mode='train', random_teaching=True, epoch=None):        # mode = train / dev / test
        if mode != 'test':
            tag_ids = batch.tag_ids.to(self.device)        #N*L
        tag_mask = batch.tag_mask.to(self.device)       #N*L
        input_ids = batch.input_ids.to(self.device)    # N*L
        lengths = batch.lengths         # N
        
        max_len = max(lengths)
        e_hiddens, h_t_c_t = self.encoder(input_ids, lengths)
        hsize = h_t_c_t[0].shape[0]
        index1, index2 = tuple(i for i in range(hsize) if i % 2 == 0) , tuple(i for i in range(hsize) if i % 2 != 0)
        h_t_c_t = ( torch.cat((h_t_c_t[0][index1, :, :], h_t_c_t[0][index2, :, :]), dim=-1), 
                    torch.cat((h_t_c_t[1][index1, :, :], h_t_c_t[1][index2, :, :]), dim=-1))
        tag_inputs = (torch.zeros((batch.size, ), dtype=torch.int) + torch.tensor([self.bos_id])).to(self.device)
        
        probs_logits = torch.empty((batch.size, max_len, self.config.num_tags)).to(self.device)  # probs_logits = [batch_szie * Length * num_tags]
        
        for i in range(max_len):
            prob, h_t_c_t = self.decoder(tag_inputs, h_t_c_t, e_hiddens, tag_mask)   # prob = [batch_size * num_tags]
            probs_logits[:, i, :] = prob             
            if (mode == 'train'):
                tag_inputs = tag_ids[:, i]                # tag_inputs = [batch_size]
                if random_teaching:
                    rand = torch.rand(1)[0]
                    if rand <= 0.25:
                        tag_inputs = torch.argmax(prob, dim=-1)
            else:
                tag_inputs = torch.argmax(prob, dim=-1)             # res = [batch_size]
        
        probs_logits += (1 - tag_mask).unsqueeze(-1).repeat(1, 1, self.config.num_tags) * -1e32   # 避免超过length的维度对softmax 产生干扰
        probs = torch.softmax(probs_logits, dim=-1)  # probs = [batch_size * Length * num_tags]  , after softmax, get the real probablity
        
        if mode == 'test':        # in test mode, we can't cal the loss , since there is no label
            return probs          
        loss = self.loss(probs_logits.view(-1, probs_logits.shape[-1]), tag_ids.view(-1))
        return probs, loss
    
    def decode(self, label_vocab, batch, mode='dev'):       #label_vocab: id2tags
        batch_size = len(batch)
        labels = batch.labels
        if mode == 'dev':
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
        if mode == 'dev':
            return predictions, labels, loss.cpu().item()
        elif mode == 'test':
            return predictions, labels
        
    
    

        


