#coding=utf8
import sys, os, time, gc
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD, BOS
from seq2seqmodel.seq2seqmodel import AttnSeq2seq

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
#train_dataset = Example.load_dataset(train_path)           # train_dataset 为 ex 的数组，每个ex为一句话有关的标签信息
test_dataset = Example.load_dataset(test_path, test=True)
#print("Dataset size: train -> %d ; test -> %d" % (len(train_dataset), len(test_dataset)))

args.vocab_size = Example.word_vocab.vocab_size         #字典大小，即有多少字
args.pad_idx = Example.word_vocab[PAD]                  #占位符<pad>对应的id，如果没有，则返回<unk>对应的id
args.num_tags = Example.label_vocab.num_tags            #标签的数目（O, B-act-slot, I-act-slot)
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)    #占位符 <pad> 对应的 id
args.tag_bos_idx = Example.label_vocab.convert_tag_to_idx(BOS)


model = AttnSeq2seq(args).to(device)
Example.word2vec.load_embeddings(model.encoder.embed, Example.word_vocab, device=device)

model.load_state_dict(torch.load(open('model.bin', 'rb'))['model'])
model.eval()
dataset = test_dataset
predictions, labels = [], []
total_loss, count = 0, 0
with torch.no_grad():
    for i in range(0, len(dataset), args.batch_size):
        cur_dataset = dataset[i: i + args.batch_size]
        current_batch = from_example_list(args, cur_dataset, device, train=False)
        pred, label, loss = model.decode(Example.label_vocab, current_batch)
        print(pred)
