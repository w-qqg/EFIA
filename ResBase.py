#!/usr/bin/env python
# coding: utf-8

# In[1]:
from nnutils.all import *
from nnutils.resnets import *
# Environment:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(torch.cuda.is_available(),device)

# Setting and Description:----------------------------------- #
bs = 80
lr = 1e-3
opt = 'Adamax'# SGD or Adamax
first_train = True  # False True
model_name = "ResTwt_lr_{}".format(str(lr))
opt = 'Adamax'
logger = initLogger(model_name)
logger.info("pretrained by ImageNet \n"+"Batch_Size ="+str(bs)+" Optimizer ="+ opt +" Lr ="+ str(lr))
latest_model_path = './ckpt/WRS-{}_latest.pth'.format(model_name)
early_stopping_COS = EarlyStopping(name=model_name, patience=15)
early_stopping_KL = EarlyStopping(name=model_name, patience=15)
epsilon = 1e-6
# L1Std = L1StdLoss()
EMD = EMDLoss()
KL = nn.KLDivLoss(reduction="batchmean")

root = './Twitter_LDL/images/'
# root = 'D:/dataset/Sentiment_LDL/Twitter_LDL/images/'

early_stopping_COS = EarlyStopping(name=model_name, patience=15)
early_stopping_KL = EarlyStopping(name=model_name, patience=15)
from tensorboardX import SummaryWriter
writerKL = SummaryWriter('./stat/{}/KL'.format(model_name), comment = model_name)
writerCOS = SummaryWriter('./stat/{}/COS'.format(model_name), comment = model_name)
writerAcc = SummaryWriter('./stat/{}/Acc'.format(model_name), comment = model_name)

class LayerOutput(nn.Module):
    def __init__(self):
        super(LayerOutput, self).__init__()
        self.fc = nn.Linear(2048, 8)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = resnet50(pretrained=True)
model.fc = LayerOutput()
    
model = model.cuda()


# In[2]:
model = nn.DataParallel(model)
torch.backends.cudnn.benchmark = True
epoch_start = 0
# if first_train:
#     latest_model_path = './ckpt/WRS-EMD_only_Adamax_4_test0.42917.pth'
#     loaded_dict = torch.load(latest_model_path)
#     model.load_state_dict(loaded_dict)
#     logger.info("Loaded CKPT:"+ latest_model_path)

#     # wrn_net_dict = (torch.load(ckpt_path))
#     # dict_trained = wrn_net_dict
#     # dict_new = model.state_dict().copy()
#     # new_list = list ( model.state_dict().keys() )
#     # trained_list = list ( dict_trained.keys() )

#     # # 6-65为Layer1；66-143为Layer2；144-257为Layer3
#     # for i in range(258): 
#     #     dict_new[ new_list[i] ] = dict_trained[ trained_list[i] ]

#     # model.load_state_dict(dict_new)
# else:
#     # epoch_start = 3
#     # latest_model_path ='./ckpt/Base_Twitter_lr_0.001_Ep4_test0.83488.pth'
#     # # latest_model_path ='C:/Users/15244/Documents/ckpt/Map-None-Red-256dx8-FC_Ep5_test0.85381.pth'
#     # loaded_dict = torch.load(latest_model_path)
#     # # trained_list = list ( loaded_dict.keys() )# len=376, 最后两个为FC的参数，可删
#     # # print("dict.keys():\n", trained_list)
#     # model.load_state_dict(loaded_dict)
#     # logger.info("Loaded CKPT:"+ latest_model_path)
#     logger.info("Not loaded")


# In[3]:

if opt=='SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
elif opt=='Adamax':
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr) 


# In[4]:

train_dataset = LDLdataset('train_processed.txt',mode="train",root=root, vote_num=8) # vote_num=11
test_dataset = LDLdataset('test_processed.txt',mode="test",root=root, vote_num=8)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs*4, shuffle=False, num_workers=0)
len(train_dataset),len(test_dataset)


# In[5]:


def train():
    model.train()
    loss_all = 0
    num_data = 1
    for data, ground, fn in train_loader:
#     for data, cls, fn in eval_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = ground.to(device)
        # Loss_Function
        loss = EMD((output),(label))# + L1Std((output),(label))*0.5

        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        if num_data==1 or num_data%200 == 0:
            print("【train】", num_data, time.strftime( "%H:%M",time.localtime() )  )
        num_data += 1
    return loss_all / len(train_dataset)


# In[6]:


def evaluate(loader,name):
    # global model
    model.eval()
    total = 0
    top1 = 0
    predictions = []
    labels = []
    kl_all = 0
    emd_all = 0
    cheb_all = 0
    itsc_all = 0
    cos_all = 0
    clark_all = 0
    canber_all = 0
    global epsilon #1e-6
    with torch.no_grad():
        for data, ground, fn in tqdm(loader):
            
            data = data.to(device)

            output = model(data)
            ground = ground.detach().cpu()
            pred = (output).detach().cpu()
            t_data_x = pred

            a = (pred)
            b = (ground)
            
            kl_all += KL_sum((a + epsilon).log(), b).item()

            emd_all += EMD(a, b).item()

            cheb_dis = abs(a-b).max(dim=1).values.sum().item()
            cheb_all += cheb_dis

            itsc_dis = torch.min(a, b).sum().item()
            itsc_all += itsc_dis
                        
            clark_all += ((a-b).pow(2)/((a+b).pow(2) +epsilon)).sum(dim=1).pow(1/2).sum().item()
            canber_all += (abs(a-b)/(a+b +epsilon)).sum().item()
            cos_all += COS((pred),(ground)).sum().item()
            
            # pred = torch.max(pred,1)[1] 
            # label = torch.max(ground,1)[1] .detach().cpu()
            # Acc：
            row = 0
            max_val_list = []
            max_val_pos = torch.max(pred,1)[1]
            gt_values = torch.max(ground,1)[0]
            for col in max_val_pos:
                max_val_list.append(ground[row,col.item()])
                row+=1
            max_vals = torch.stack(max_val_list, 0)
            correct_nums = (max_vals == gt_values).sum().item()
            top1 += correct_nums
            # _,max2 = torch.topk(t_data_x,2,dim=-1) # origin pred

            # # total += label.size(0)
            # label = label.view(-1,1)
            # top1 += (label == max2[:,0:1]).sum().item()
            
    # predictions = np.hstack(predictions)
    # labels = np.hstack(labels)
    total = len(loader.dataset)
    Acc = top1/total
    meanKL = kl_all/total
    meanEDM = emd_all/total
    meanCOS = cos_all/total
    meanCheb = cheb_all/total
    meanItsc = itsc_all/total
    meanClark = clark_all/total
    meanCanber = canber_all/total
#     print("predictions,labels",predictions,labels)
    # print("【Eval_" + name + "】topK Acc:", Acc,  time.strftime( "%H:%M",time.localtime() ))
    logger.info("an epoch evaluated：\nAcc = %s, Cheb = %s, Clark = %s, Canber = %s, ", str(Acc), str(meanCheb), str(meanClark), str(meanCanber))
    logger.info("KLdiv = %s, Cosine = %s, Itsc = %s", str(meanKL), str(meanCOS), str(meanItsc))
    # print("Cheb:(<0.25)",meanCheb)
    # print("Clark:(<2.2)",meanClark)
    # print("Canber:(<5.5)",meanCanber)
    # print("KLdiv:(<0.45)",meanKL)
    # print("Cosine:(>0.84)",meanCOS)
    # print("Itsc:(>0.6)",meanItsc)

    return top1/total, meanCOS, meanKL

# In[7]:


# In[8]:
# 提前测试
test_acc, test_COS, test_KL = evaluate(test_loader,"test_loader")
print('Epoch0, Test Auc: {:.5f}, Test COS: {:.5f}, Test KL: {:.5f}'.
          format(test_acc, test_COS, test_KL))
# 开始训练
for e in range(epoch_start,100):
    epoch_start += 1
    logger.info("an epoch started: %s", str(epoch_start))
    loss = train()
    torch.save(model.state_dict(), latest_model_path)
    # scheduler.step()
    logger.info("an epoch trained：loss = %s", str(loss))

    # train_acc, train_COS, train_KL = evaluate(train_loader,"train_loader")
    test_acc, test_COS, test_KL = evaluate(test_loader,"test_loader")
    
    writerKL.add_scalar(model_name, test_KL, global_step=e)
    writerCOS.add_scalar(model_name, test_COS, global_step=e)
    writerAcc.add_scalar(model_name, test_acc, global_step=e)

    logger.info("an epoch evaluated：\ntest_acc = %s, test_COS = %s, test_KL = %s",str(test_acc),str(test_COS),str(test_KL))
    print('Epoch: {:03d}, Loss: {:.5f} , Test Auc: {:.5f}, \nTest COS: {:.5f}, Test KL: {:.5f}'.
          format(epoch_start, loss, test_acc, test_COS, test_KL))
    
    if epoch_start>5: early_stopping_KL(epoch_start, test_KL, model, tend='inverse')
    early_stopping_COS(epoch_start, test_COS, model, tend='direct')
    if early_stopping_COS.early_stop:
        print("Early stopping")
        logger.info("Early stopping")
        break

# In[ ]:
train_acc,train_COS,train_KL = evaluate(train_loader,"train_loader")
test_acc,test_COS,test_KL = evaluate(test_loader,"test_loader")
print(train_acc,train_COS,train_KL)
print(test_acc,test_COS,test_KL)


# In[10]:

# torch.save(model.state_dict(), './ckpt/WRS-(EMD+L1Std)_epoch18_cos709_KLinf.pth')
# logger.info("model saved")
