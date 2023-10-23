import tqdm
import argparse
from utils.config import Config
from torch.autograd import Variable
import torch
import data_loader
import time
import sys
from data_loader import ForeverDataIterator
from torch.optim.lr_scheduler import LambdaLR
import wandb

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer

 
#==============eval
def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        probabilities = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])
    #accuracy = torch.tensor.cpu(torch.sum(torch.squeeze(predict).float() == all_labels)).numpy() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state)
    return {'accuracy':accuracy}


def train(model_instance, train_source1_iter, train_source2_iter, train_source3_iter, test_target_loader,
          group_ratios1,group_ratios4, max_iter, optimizer1,optimizer4, lr_scheduler, eval_interval,checkpoint_=False,load_path=None):
    model_instance.set_train(True)
    print("start train...")
    iter_num = 0
    epoch = 0
    if checkpoint_:
        print("true")
        checkpoint = torch.load(load_path) 
        epoch = checkpoint['epoch']
        iter_num = checkpoint['iter_num']
    total_progress_bar = tqdm.tqdm(desc='Train iter',initial =iter_num, total=max_iter,file=sys.stdout)
    print("Start Training")
    best_acc=0
    while True:
        for i in range(220):
            inputs_source_1, labels_source_1 = next(train_source1_iter)[:2]
            inputs_source_2, labels_source_2 = next(train_source2_iter)[:2]
            inputs_source_3, labels_source_3 = next(train_source3_iter)[:2]


            optimizer1 = lr_scheduler.next_optimizer(group_ratios1, optimizer1, iter_num/5)
            optimizer1.zero_grad()

            optimizer4 = lr_scheduler.next_optimizer(group_ratios4, optimizer4, iter_num/5)
            optimizer4.zero_grad()

            
            if model_instance.use_gpu:
                inputs_source_1, inputs_source_2, inputs_source_3,  labels_source_1, labels_source_2, labels_source_3 = Variable(inputs_source_1).cuda(), Variable(inputs_source_2).cuda(),Variable(inputs_source_3).cuda(), Variable(labels_source_1).cuda(), Variable(labels_source_2).cuda(), Variable(labels_source_3).cuda()
            else:
                inputs_source_1, inputs_source_2, inputs_source_3,  labels_source_1, labels_source_2, labels_source_3 = Variable(inputs_source_1), Variable(labels_source_1), Variable(inputs_source_2),Variable(inputs_source_3), Variable(labels_source_2), Variable(labels_source_3)

            train_batch(model_instance, inputs_source_1, inputs_source_2, inputs_source_3, labels_source_1, labels_source_2, labels_source_3, optimizer1,optimizer4)

            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result = evaluate(model_instance, test_target_loader)
                if eval_result['accuracy'] > best_acc:
                    best_acc = eval_result['accuracy']
                time.sleep(1)
            iter_num += 1
            total_progress_bar.update(1)
        tqdm.tqdm.write("Epoch "+str(epoch)+" Done. Iter num: "+str(iter_num))
        epoch += 1
        if iter_num >= max_iter:
            break
    print('finish train')
    print('Target Accuracy: '+str(best_acc))

def train_batch(model_instance, inputs_source_1, inputs_source_2, inputs_source_3, labels_source_1, labels_source_2, labels_source_3, optimizer1,optimizer4):
    inputs = torch.cat((inputs_source_1, inputs_source_2, inputs_source_3), dim=0)
    model_instance.get_loss(inputs, labels_source_1, labels_source_2, labels_source_3,optimizer1,optimizer4)
 
if __name__ == '__main__':
    from model.MDD import MDD

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='address to dann.yml file in config folder')
    parser.add_argument('--dataset', default='Office-Home', type=str,
                        help='which dataset')
    args = parser.parse_args()

    cfg = Config(args.config)

    checkpoint_ = False
    load_path = "Mention path"

    if args.dataset == 'Office-Home':
        class_num = 65 
        width = 2048 
        srcweight = 2 # margin
        is_cen = False

    else:
        width = -1
    

    model_instance = MDD(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight,checkpoint_=checkpoint_,load_path = load_path)

    batch_size = 32 
    root_path = 'enter root path'
    source1_name ='Art' 
    source2_name ='Clipart' 
    source3_name = 'Product'

    target_name ='Real World' 

    train_source1_loader = data_loader.load_data(root_path, source1_name, batch_size)
    train_source2_loader = data_loader.load_data(root_path, source2_name, batch_size)
    train_source3_loader = data_loader.load_data(root_path, source3_name, batch_size)

    test_target_loader = data_loader.load_data(root_path, target_name, batch_size)

    print(len(train_source1_loader))
    print(len(train_source2_loader))
    print(len(train_source3_loader))
    print(len(test_target_loader))

    train_source1_iter = ForeverDataIterator(train_source1_loader)
    train_source2_iter = ForeverDataIterator(train_source2_loader)
    train_source3_iter = ForeverDataIterator(train_source3_loader)

    param_groups1 = model_instance.get_parameter_list(flag=1)
    group_ratios1 = [group['lr'] for group in param_groups1]

    param_groups4 = model_instance.get_parameter_list(flag=4)
    group_ratios4 = [group['lr'] for group in param_groups4]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer1 = torch.optim.SGD(param_groups1, **cfg.optim.params)

    optimizer4 = torch.optim.SGD(param_groups4,**cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    
    if checkpoint_:
        print("true")
        checkpoint = torch.load(load_path)

    train(model_instance, train_source1_iter, train_source2_iter, train_source3_iter, test_target_loader, group_ratios1, group_ratios4,
          max_iter=2000, optimizer1=optimizer1, optimizer4=optimizer4, lr_scheduler=lr_scheduler, eval_interval=220,checkpoint_=checkpoint_,load_path=load_path)
