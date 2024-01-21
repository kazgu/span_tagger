import os
try:
    import torch
except:
    print('instaling torch')
    os.system("pip install torch")

try:
    from tqdm import tqdm
except:
    print('instaling tqdm')
    os.system("pip install tqdm")

try:
    import requests
except:
    print('instaling requests')
    os.system("pip install requests")

try:
    from torchcrf import CRF
except:
    print('instaling pytorch-crf')
    os.system("pip install pytorch-crf")

from train import *
from my_args import *
from dataset import *
from model.BiLSTM_SPAN import BiLSTM_SPAN
import requests
import json
set_seed(10000)
def send_model(args,save_path,save_path_args,test_acc,avg_loss):
    print('send_model........')
    server_addres=args.server_addres
    model = open(save_path, 'rb')
    file = {'file':model}
    if int(args.ClientID)==1:
        item={'text_data2id':args.text_field.data2id,'label_data2id':args.label_field.data2id}
        torch.save(item,'%s_a'%save_path_args)
        file['args']=open('%s_a'%save_path_args, 'rb')
    data = {'taskid':args.TaskID,'clientid':args.ClientID,'acc':test_acc,'loss':avg_loss}
    req = requests.post(url='%s/send_model'%server_addres,data=data, files=file)
    print(req.text)

def modeltrain(bast_acc):
    args = build_args_parser()
    model=BiLSTM_SPAN
    print('Loading data Iterator ...')
    text_field, label_field = create_field(args)
    train_iter, dev_iter, test_iter = load_dataset(text_field, label_field,args)
    args.text_field=text_field
    args.label_field=label_field
    args.vocabulary_size = len(text_field.data2id)
    args.vocabulary_size_trg = len(label_field.data2id)
    args.class_num = len(label_field.data2id)
    # args.cuda = args.device != -1 and torch.cuda.is_available()
    args.cuda =torch.cuda.is_available()
    args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
    print('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        if attr in {'vectors'}:
            continue
        print('\t{}={}'.format(attr.upper(), value))
    net = model(args)
    if args.isContinueTrain:
        save_prefix = os.path.join(args.save_dir, '%s_%s'%(args.TaskID,args.ClientID))
        save_path = '{}_model.cpth'.format(save_prefix)
        if os.path.exists(save_path):
            print('isContinueTrain','*'*30,save_path)
            print('\nLoading model from {}...\n'.format(save_path))
            net.load_state_dict(torch.load(save_path))
    if args.cuda:
        device = torch.device("cuda:%s"%(args.device)) 
        torch.cuda.set_device(device)
        net = net.cuda()
    if args.isTrain:
        try:
            bast_acc=train(train_iter, dev_iter, net, args,bast_acc)
        except KeyboardInterrupt:
            print('Exiting from training early')
    print('*'*30 + ' Testing ' + '*'*30)
    save_prefix = os.path.join(args.save_dir, '%s_%s'%(args.TaskID,args.ClientID))
    save_path = '{}_model.cpth'.format(save_prefix)
    save_path_args = '{}_model.arg'.format(save_prefix)
    state_dict = torch.load(os.path.join(save_path))
    net.load_state_dict(state_dict)
    test_acc,avg_loss = test(test_iter, net, args)
    #############################################
    send_model(args,save_path,save_path_args,test_acc,avg_loss)
    return bast_acc

def get_status():
    args = build_args_parser()
    save_prefix = os.path.join(args.save_dir, '%s_%s'%(args.TaskID,args.ClientID))
    save_path = '{}_model.cpth'.format(save_prefix)
    server_addres=args.server_addres
    data = {'taskid':args.TaskID,'clientid':args.ClientID}
    req = requests.post(url='%s/get_status'%server_addres,data=data)
    result=req.json()['result']
    if result==3 and req.json()['modelurl']:
        durl='%s%s'%(server_addres,req.json()['modelurl'])
        if not os.path.exists(save_prefix):
            os.mkdir(save_prefix)
        r=requests.get(durl)
        print('downloading server model.......')
        with open(save_path,"wb") as f:
            f.write(r.content)
    print(req.json())
    return result,req.json()['clienttrain']
    

if __name__ == '__main__':
    status,clienttrain=get_status()
    bast_acc=0
    while int(status)!=2:
        try:
            if int(status)!=2 and clienttrain==0:
                bast_acc=modeltrain(bast_acc)
            status,clienttrain=get_status()
        except Exception as ex:
            print(ex)
        time.sleep(5)