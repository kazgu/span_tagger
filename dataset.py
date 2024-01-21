from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch
import random
pdir=os.path.dirname(os.path.abspath(__file__))
class DataFeilds():
    def __init__(self,args,index,sequential=False,isText=True,fix_length=None,tokenize=None,public_vocab=False,pads=[]):
        self.args=args
        self.index=index
        self.data_list=[]
        self.sequential=sequential
        self.isText=isText
        self.fix_length=fix_length
        self.tokenize=tokenize
        self.pads=pads
        self.public_vocab=public_vocab
        self.data2id={}
        self.id2data={}
        if public_vocab:
            self.data2id=torch.load(os.path.join(pdir,'word2id.w'))
            self.id2data={self.data2id[k]:k for k in self.data2id }
    def add_data(self,data_list):
        for data in data_list:
            if type(data[self.index]) is list:
                for ddt in data[self.index]:
                    self.data_list.append(ddt)
            else:
                self.data_list.append(data[self.index])
    def build_vocab(self,*datas):
        if not self.public_vocab:
            if datas:
                for data in datas:
                    self.add_data(data)
            vocab=self.pads+sorted(list(set(self.data_list)))
            self.id2data={i:w for i,w in enumerate(vocab)}
            self.data2id={w:i for i,w in enumerate(vocab)}
    def tokenize2id(self,args,data_list):
        data2ix_list = []
        org_data=[]
        sen_len=args.sen_len if args.sen_len>0  else max([len(se[self.index]) for se in data_list])
        # if sen_len>510:
        #     sen_len=510
        def padding(dlist):
            if len(dlist)>sen_len:
                return dlist[:sen_len]
            return dlist+[0 for i in range(sen_len-len(dlist))]
        for i in tqdm(range(len(data_list))):
            if self.sequential:
                data2ix_list.append(padding([self.data2id.get(w,self.data2id['[UNK]']) for w in data_list[i][self.index]] ))
            else:
                data2ix_list.append([self.data2id.get(w,self.data2id['[UNK]']) for w in [data_list[i][self.index]]])
            org_data.append(''.join([w for w in data_list[i][self.index]][:sen_len]))
        return  data2ix_list,org_data

def tokenizer(x):
    return [w for w in x]
 
def create_field(args):
    text_field = DataFeilds(args,index=0,sequential=True, fix_length=args.sen_len,public_vocab=args.public_vocab,pads=['[PAD]','[UNK]'] )
    label_field =DataFeilds(args,index=1,sequential=True, fix_length=args.sen_len,pads=['[PAD]','[UNK]'])
    return text_field, label_field


def get_dataset(args):
    files=['train.tsv','valid.tsv','test.tsv']
    def getdata(args,file):
        data=[]
        if args.readtype==1:
            with open(os.path.join(args.dataset,file),encoding='utf8') as f:
                lines=f.readlines()
                for line in tqdm(lines):
                    if line.strip():
                        line=line.strip().split('\t')
                        if len(line)==2:
                            words,label=line
                            if words and label:
                                data.append([words,label])
        else:
            with open(os.path.join(args.dataset,file),encoding='utf8') as f:
                lines=f.readlines()
                text=[]
                labels=[]
                for line in tqdm(lines):
                    if line.strip():
                        line=line.strip().split('\t')
                        if len(line)==2:
                            word,label=line
                            if word and label:
                                text.append(word.strip())
                                labels.append(label.strip())
                    else:
                        if len(text)==len(labels) and len(text)>0:
                            data.append([text,labels])
                            text=[]
                            labels=[]


        return data
                
    train, valid, test = getdata(args,files[0]),getdata(args,files[1]),getdata(args,files[2])
    return train, valid, test

 
def load_dataset(text_field, label_field, args):
    # ************************** get torch text dataset ***************************
    train_dataset, dev_dataset, test_dataset = get_dataset(args)

    # ************************** build vocabulary *********************************
    text_field.build_vocab(train_dataset, dev_dataset)  # build vocab from train/val dataset only

    label_field.build_vocab(train_dataset, dev_dataset)  # change from '0', '1' to 0,1

    print('Num of class is ' + str(len(label_field.data2id)))
    print('Num of words is ' + str(len(text_field.data2id)))

    def build_dataset(datalist):
        ret=[]
        text_data,org_data=text_field.tokenize2id(args,datalist)
        label_data,_=label_field.tokenize2id(args,datalist)
        for t,l,o in zip(text_data,label_data,org_data):
            ret.append([torch.tensor(t),torch.tensor(l),o])
        return ret
    # **************************  build Iterator ***********************************
    train_iter = DataLoader(MyDataset(build_dataset(train_dataset)), batch_size=args.batch_size, shuffle=True)
    dev_iter = DataLoader(MyDataset(build_dataset(dev_dataset)), batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(MyDataset(build_dataset(test_dataset)), batch_size=args.batch_size, shuffle=False)
    return train_iter, dev_iter, test_iter 

def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data_dt):
        super(MyDataset, self).__init__()
        self.data=data_dt
    def __getitem__(self, index):
        data_ix=self.data[index][0]
        tags_ix=self.data[index][1]
        data_org=self.data[index][2]
        return data_ix,tags_ix,data_org
 
    def __len__(self):
        return len(self.data)