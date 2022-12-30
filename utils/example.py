import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path) #根据对话中的语句里的字构建词表
        cls.word2vec = Word2vecUtils(word2vec_path)                 
        cls.label_vocab = LabelVocab(root)              #tag('B/I-act-slot') 对应的 id

    @classmethod
    def load_dataset(cls, data_path, test=False):         # examples 是 ex 的数组，每个 ex 都包含了一句话里 act-slot-value 以及对应 tag 的信息
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt, test)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, test):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']                #utt是语音转换成的语句
        self.input_idx = [Example.word_vocab[c] for c in self.utt]                #self.input_idx 即将语句里的字转换成id的列表
        if test:
            return
        self.slot = {}                             # self.slot 是 这句话里 {act-slot : value} 的字典，value 就是语句里的词
        for label in ex['semantic']:             #ex['semantic']是 这句话里 act-slot-value 的列表
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)        #self.tags 是这句话里每个字的标签
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)          # 找到 value 在这句话中的起始位置
            if bidx != -1:                                                            # 然后将 self.tags 对应value的部分打上 B, I 的标签
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)      
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]    #self.slotvalue 是 'act-slot-value' 的列表
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]           # 将tags里的标签转换成对应的id
