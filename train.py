import os
import time
import torch
import requests
import torch.nn.functional as F

def send_status(args,valid_accs,avg_losss,process):
    valid_acc=sum(valid_accs)/len(valid_accs)
    avg_loss=sum(avg_losss)/len(avg_losss)
    print('send_status',valid_acc,avg_loss,process)
    server_addres=args.server_addres
    data = {'taskid':args.TaskID,'clientid':args.ClientID,'acc':valid_acc,'loss':avg_loss,'process':process}
    req = requests.post(url='%s/send_status'%server_addres,data=data)
    print(req.text)

def metric(args,logits,target):
    logits,target=logits.cuda(),target.cuda()
    if args.readtype==1:
        allsize=target.shape[0]*target.shape[1] if len(target.shape)>1 else target.shape[0]
        logits=torch.max(logits, 1)[1]
        correct = (logits==target.view(-1)).sum()
        acc=correct.item()/allsize
        f1=acc
        return acc,f1,correct
    else:
        logits=torch.max(logits, 1)[1] if len(logits.shape)>2 else logits
        no=args.label_field.data2id['O']
        pad=args.label_field.data2id['[PAD]']
        unk=args.label_field.data2id['[UNK]']
        correct=0
        allsize=0
        found=0
        origin=0
        matched=0
        for l,t in zip(logits,target):
            reallen=len(t)-((t==pad).sum())
            l=l[:reallen]
            t=t[:reallen]
            allsize+=reallen
            origin+=reallen-((t==no).sum())
            found+=reallen-((l==pad).sum())-((l==unk).sum())-((l==no).sum())
            correct+=sum([1 for _t,_l in zip(t,l) if _t==_l and _t.item()!=no ])
            matched+=((t==l).sum())
        recall = 0 if origin == 0 else (correct / origin.item())
        precision = 0 if found == 0 else (correct / found.item())
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return (matched/allsize).item(),f1,correct
        
        

def train(train_iter, dev_iter, model, args,best_acc=0):
    if args.cuda:
        model.cuda()
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = 0
    # best_acc = 0
    last_step = 0
    lrupdatetimes=10
    canlrupdate=False
    epoch_avg_acc=[]
    epoch_avg_loss=[]
    epoch_process=[]
    args.test_interval=(len(train_iter.dataset)/args.batch_size)//2
    model.train() 
    for epoch in range(1, args.epochs + 1):
        epoch_process.append(epoch)
        for feature,target,_ in train_iter:
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits,target,loss = model(feature,target)
            epoch_avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                train_acc,f1,corrects=metric(args,logits,target)
                print('Epoch {} ,Iteration:[{}/{}] - loss: {:.6f}  acc: {:.4f} f1:{:.4f} ,%({}/{})'.format(epoch,steps,len(train_iter.dataset)//args.batch_size,
                                                                             loss.item(),
                                                                             train_acc*100,
                                                                             f1*100,
                                                                             corrects,
                                                                             args.batch_size))
 
            if canlrupdate:
                canlrupdate=False
                lrupdatetimes-=1
                learning_rate /= 2
                update_lr(optimizer, learning_rate)

            if steps % args.test_interval == 0:
                dev_acc = evaluation(dev_iter, model, args)
                epoch_avg_acc.append(dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args)
                else:
                    if steps - last_step >= args.early_stopping:
                        if lrupdatetimes>0:
                            canlrupdate=True
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        # raise KeyboardInterrupt
                model.train()
                send_status(args,epoch_avg_acc,epoch_avg_loss,int((epoch/args.epochs)*100))
        if args.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return best_acc 

 
def evaluation(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    accs=0
    size=0
    f1s=0
    with torch.no_grad():
        for feature,target,_ in data_iter:
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            logits,target,loss = model(feature,target)
            avg_loss += loss.item()
            acc,f1,correct=metric(args,logits,target)
            print('Loss:%.3f,Acc:%.3f,F1:%.3f,Correct:%s'%(loss.item(),acc,f1,correct))
            accs+=acc
            f1s+=f1
            corrects+=correct
            size+=1
    avg_loss /= size
    accuracy = accs/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f} f1:{:.4f} %({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       f1s/size,
                                                                       corrects,
                                                                       size))
    return accuracy

def test(data_iter, model, args):
    model.eval()
    corrects, avg_loss ,avf1= 0, 0,0
    time_start = time.time()
    isize=0
    size=0
    accs=0
    f1s=0
    with torch.no_grad():
        for feature,target,_ in data_iter:
            isize+=1
            if args.cuda:
                feature, target= feature.cuda(), target.cuda()
            logits,target,loss = model(feature,target)
            avg_loss += loss.item()
            acc,f1,correct=metric(args,logits,target)
            print('Loss:%.3f,Acc:%.3f,F1:%.3f,Correct:%s'%(loss.item(),acc,f1,correct))
            accs+=acc
            f1s+=f1
            corrects+=correct
            size+=1
    time_end = time.time()
    print('Test total cost {:.4f} s'.format(time_end - time_start))

    avg_loss /= size
    accuracy = accs/size
    print('\nTest - loss: {:.6f}  acc: {:.4f} f1:{:.4f} %({}/{})\n'.format(avg_loss,
                                                                       accuracy,
                                                                       f1s/size,
                                                                       corrects,
                                                                       size))
    return accuracy,avg_loss

 
def save(model, args):
    save_dir=args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, '%s_%s'%(args.TaskID,args.ClientID))
    save_path = '{}_model.cpth'.format(save_prefix)
    args_save_path = '{}_model.arg'.format(save_prefix)
    torch.save(model.state_dict(), save_path)
    torch.save(args, args_save_path)


# For updating the learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr