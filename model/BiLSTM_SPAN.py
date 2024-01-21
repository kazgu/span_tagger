import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torchcrf import CRF
torch.manual_seed(10000)
class BiLSTM_SPAN(nn.Module):
    def __init__(self, args):
        super(BiLSTM_SPAN, self).__init__()

        class_num = args.class_num    # 3, 0 for unk, 1 for negative, 2 for postive
        vocabulary_size = args.vocabulary_size  # total number of vocab (2593)
        embedding_dimension = args.embedding_dim     # 128
        hidden_size = args.hidden_size
        self.args = args
        self.entity_dict={} 
        for lb in self.args.label_field.data2id:
            if lb.startswith('I') or lb.startswith('I_'):
                continue
            ent=lb.replace('B-','').replace('B_','')
            if ent not in self.entity_dict:
                self.entity_dict[ent]=[self.args.label_field.data2id[lb]]
            else:
                self.entity_dict[ent]+=[self.args.label_field.data2id[lb]]
        entity_num=[self.entity_dict[ent][0] for ent in self.entity_dict]
        self.new_enti2id={ed:i for i,ed in enumerate(entity_num) }
        self.id2new_enti={i:ed for i,ed in enumerate(entity_num) }
        # Embedding Layer
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        self.bilstm1 = nn.LSTM(input_size=self.args.embedding_dim,
                            hidden_size=self.args.hidden_size,
                            num_layers=self.args.hidden_layers,#self.args.hidden_layers//2 if self.args.hidden_layers//2>=1 else 1
                            dropout=self.args.dropout,
                            bidirectional=True,
                            batch_first=True
                            ) 
        self.bilstm2 = nn.LSTM(input_size=self.args.hidden_size,
                            hidden_size=self.args.hidden_size,
                            num_layers=self.args.hidden_layers,
                            dropout=self.args.dropout,
                            bidirectional=True,
                            batch_first=True
                            )

        self.net2=nn.Sequential(
            nn.Linear(self.args.hidden_size,self.args.hidden_size),
            nn.ReLU()
        )


        self.dropout = nn.Dropout(args.dropout) 
        self.dropout2 = nn.Dropout(args.dropout) 
        self.fc1 = nn.Linear(args.hidden_size, 4)

        
        self.fc2 = nn.Linear(args.hidden_size, len(self.new_enti2id))


        # self.fc_cls = nn.Linear(args.hidden_size, 2)
        self.W = nn.Linear(self.args.embedding_dim+2 * args.hidden_size, args.hidden_size)
        self.W2 = nn.Linear( args.hidden_size*3, args.hidden_size)

        self.ln1 = nn.LayerNorm(self.args.embedding_dim+2 * args.hidden_size)
        self.ln2 = nn.LayerNorm(args.hidden_size*3)

        self.tanh = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.relu2=nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()

        self.softmax=nn.Softmax(dim=-1)
        self.softmax2=nn.Softmax(dim=-1)
    def detect_entity_forward(self, x,y=None):
        x=self.embedding(x)
        lstm_out, _ = self.bilstm1(x)
        input_features = self.ln1(torch.cat([lstm_out, x], dim=2))
        linear_output = self.tanh(self.W(input_features))
        # linear_output = self.dropout(linear_output)
        out_liner=torch.cat([lstm_out, linear_output], dim=2)
        final_out = self.fc1(self.relu(linear_output))
        # print('final_out.shape',final_out.shape)
        scores=self.softmax(final_out)
        scores=torch.max(scores, 2)[1] if len(scores.shape)>2 else scores
        if y is None:
            return scores,linear_output
        yen=deepcopy(y)
        # print('self.args.label_field.data2id',self.args.label_field.data2id)
        for lb in self.args.label_field.data2id:
            if lb.startswith('B'):
                yen=torch.where((yen==self.args.label_field.data2id[lb]),1,yen)
            elif lb.startswith('I'):
                yen=torch.where((yen==self.args.label_field.data2id[lb]),2,yen)
            elif lb=='[CLS]' or lb=='[SEP]':
                yen=torch.where((yen==self.args.label_field.data2id[lb]),3,yen)
            else:
                yen=torch.where((yen==self.args.label_field.data2id[lb]),0,yen)

        # print(final_out.shape,yen.shape)
        loss1= self.criterion(final_out.permute(0,2,1),yen)
        # print('loss1',loss1)
        return scores,linear_output,yen,loss1

    def entity_cls_forward(self,inputx,spany,orgy=None,test=False):
            # print('spany',spany)
            mquery=torch.where((spany==1) | (spany==2) |(spany==3))
            # print('mquery',mquery)
            ddict={}
            for row,val in zip(*mquery):
                row,val=row.item(),val.item()
                if row not in ddict:
                    ddict[row]=[[val]]
                else:
                    if ddict[row][-1][-1]+1==val:
                        ddict[row][-1]+=[val]
                    else:
                        ddict[row]+=[[val]]
            for iit in range(spany.shape[0]):
                if iit not in ddict:
                    ddict[iit]=[[]]
            maxlen=max([len(ddict[k]) for k in ddict])
            newx=[]
            newy=[]
            for k in range(spany.shape[0]):
                tmp=[]
                tmp1=[]
                for vv in ddict[k]:
                    if vv:
                        vt=inputx[k][vv].max(dim=0)[0].unsqueeze(0)
                        if not test:
                            vtu=torch.tensor([self.new_enti2id.get(orgy[k][vv][0].item(),0)],dtype=torch.long).unsqueeze(0)
                            tmp1.append(vtu)
                        tmp.append(vt)
                if not test:
                    if tmp1:
                        tmp1+=[torch.tensor([0]).unsqueeze(0) for _ in range(maxlen-len(tmp1))]
                        newy.append(torch.cat(tmp1,0).unsqueeze(0))
                        tmp+=[torch.rand((1,inputx.shape[-1])).cuda() for _ in range(maxlen-len(tmp))]
                        newx.append(torch.cat(tmp,0).unsqueeze(0))
                else:
                    tmp+=[torch.rand((1,inputx.shape[-1])).cuda() for _ in range(maxlen-len(tmp))]
                    newx.append(torch.cat(tmp,0).unsqueeze(0))
            newx=torch.cat(newx,dim=0)
            if not test:
                newy=torch.cat(newy,dim=0).squeeze(2)

            lstm_out2, _ = self.bilstm2(newx)
            # net2out=self.net2(newx)
            input_features = self.ln2(torch.cat([lstm_out2, newx], dim=2))
            # print(input_features.shape)
            linear_output2 = self.tanh2(self.W2(input_features))
            # linear_output2 = self.dropout2(linear_output2)
            final_out2 = self.fc2(self.relu2(linear_output2))

            logits=self.softmax2(final_out2)
            logits_prob=logits
            logits=torch.max(logits, 2)[1] if len(logits.shape)>2 else logits
            if not test:
                loss2= self.criterion(final_out2.cuda().permute(0,2,1),newy.cuda())
                # print('loss2',loss2)
            else:
                result=[]
                for k in range(spany.shape[0]):
                    item=[]
                    for it,iv in enumerate(ddict[k]):
                        if iv:
                            item.append([iv,self.id2new_enti[logits[k][it].item()],logits_prob[k][it]])
                    result.append(item)
                return result
            return logits,newy,loss2
    def forward(self, x,y):
        detect_scores_logits,linear_output,yen,loss1=self.detect_entity_forward(x,y)
        scores2,newy,loss2=self.entity_cls_forward(linear_output,yen,y)
        loss=loss1+loss2
        # print('scores2,loss',scores2,loss)
        return scores2,newy,loss
    def get_result(self,x,_org):
        detect_scores_logits,linear_output=self.detect_entity_forward(x,org=_org)
        clsscores=self.entity_cls_forward(linear_output,detect_scores_logits,test=True)
        return clsscores