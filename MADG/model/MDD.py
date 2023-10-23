import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
import math

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        ctx.iter_num = iter_num
        ctx.alpha = alpha
        ctx.low_value = low_value
        ctx.high_value = high_value
        ctx.max_iter = max_iter
        ctx.iter_num += 1
        output = input * 1.0
        return output
   
    @staticmethod
    def backward(ctx, grad_output):
        ctx.coeff = np.float(
            2.0 * (ctx.high_value - ctx.low_value) / (1.0 + np.exp(-ctx.alpha * ctx.iter_num / ctx.max_iter)) - (
                        ctx.high_value - ctx.low_value) + ctx.low_value)
        return -ctx.coeff * grad_output


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        self.inplace=False
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer_1 = GradientReverseLayer()
        self.grl_layer_2 = GradientReverseLayer()
        self.grl_layer_3 = GradientReverseLayer()

        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        
        self.classifier_layer_2_1_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2_1 = nn.Sequential(*self.classifier_layer_2_1_list)
        
        
        self.classifier_layer_2_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5), 
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2_2 = nn.Sequential(*self.classifier_layer_2_2_list)
        
        self.classifier_layer_2_3_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5), 
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2_3 = nn.Sequential(*self.classifier_layer_2_3_list)

        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2_1[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2_1[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer_2_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer_2_3[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2_3[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list1 = [{"params":self.base_network.parameters(), "lr":0.2},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2_1.parameters(), "lr":1},{"params":self.classifier_layer_2_2.parameters(), "lr":1},{"params":self.classifier_layer_2_3.parameters(), "lr":1}]

        self.parameter_list4 = [{"params":self.base_network.parameters(), "lr":0.2},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                        {"params":self.classifier_layer.parameters(), "lr":1}]

    def forward(self, inputs, size_1, size_2, size_3):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        
        features_1 = features[:size_1]
        features_2 = features[size_1:size_1+size_2]
        features_3 = features[size_1+size_2:size_1+size_2+size_3]

        features_adv = self.grl_layer_1.apply(torch.cat((features_1,features_2),dim=0)) 
        outputs_adv_1 = self.classifier_layer_2_1(features_adv)
        
        features_adv_2 = self.grl_layer_2.apply(torch.cat((features_1,features_3),dim=0))
        outputs_adv_2 = self.classifier_layer_2_2(features_adv_2)

        features_adv_3 = self.grl_layer_3.apply(torch.cat((features_2,features_3),dim=0))
        outputs_adv_3 = self.classifier_layer_2_3(features_adv_3)

        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv_1, outputs_adv_2, outputs_adv_3
   
    def inference(self,inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)
        return softmax_outputs

class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3,checkpoint_=False,load_path=None):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)
        
        if checkpoint_:
            checkpoint = torch.load(load_path)
            self.c_net.load_state_dict(checkpoint['model_state_dict'])
        
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight
        self.clf_wt =1.0
        self.tf_wt=1.0

    def get_loss(self, inputs, labels_source_1, labels_source_2, labels_source_3,optimizer1,optimizer4):
        class_criterion = nn.CrossEntropyLoss()
        size_1 = labels_source_1.size(0)
        size_2 = labels_source_2.size(0)
        size_3 = labels_source_3.size(0)
        
        offset = 1e-6 #torch.Tensor(1e-6)
        _, outputs, _, outputs_adv_1,outputs_adv_2,outputs_adv_3 = self.c_net(inputs, size_1, size_2,size_3)

        classifier_loss_1 = class_criterion(outputs.narrow(0, 0, size_1), labels_source_1)
        classifier_loss_2 = class_criterion(outputs.narrow(0, size_1, size_2), labels_source_2) 
        classifier_loss_3 = class_criterion(outputs.narrow(0, size_1+size_2, size_3), labels_source_3)

        total_loss = classifier_loss_1 +classifier_loss_2 +classifier_loss_3

        total_loss.backward()
        optimizer4.step()

        _, outputs, _, outputs_adv_1,outputs_adv_2,outputs_adv_3 = self.c_net(inputs, size_1, size_2,size_3)

        target_adv = outputs.max(1)[1]
        target_adv_src_1 = target_adv.narrow(0, 0, size_1)
        target_adv_src_2 = target_adv.narrow(0, size_1, size_2)
        target_adv_src_3 = target_adv.narrow(0, size_1+size_2, size_3)
   
        
        classifier_loss_adv_src_1 = -class_criterion(outputs_adv_1.narrow(0, 0, size_1), target_adv_src_1)

        x_2 = 1 - F.softmax(outputs_adv_1.narrow(0,  size_1, size_2), dim = 1)
        logloss_tgt_2 = torch.log(torch.clamp(x_2 + offset, max=1.)) 
        classifier_loss_adv_tgt_2 = -F.nll_loss(logloss_tgt_2, target_adv_src_2)

        transfer_loss_1 = -(self.srcweight * classifier_loss_adv_src_1 + classifier_loss_adv_tgt_2)

        classifier_loss_adv_src_2 = -class_criterion(outputs_adv_2.narrow(0, 0, size_1), target_adv_src_1)
        
        x_3 = 1 - F.softmax(outputs_adv_2.narrow(0,  size_1, size_3), dim = 1)
        logloss_tgt_3 = torch.log(torch.clamp(x_3 + offset, max=1.))
        classifier_loss_adv_tgt_3 = -F.nll_loss(logloss_tgt_3, target_adv_src_3)
        
        transfer_loss_2 = -(self.srcweight * classifier_loss_adv_src_2 + classifier_loss_adv_tgt_3)

        classifier_loss_adv_src_3 = -class_criterion(outputs_adv_3.narrow(0, 0, size_2), target_adv_src_2)
        
        x_4 = 1 - F.softmax(outputs_adv_3.narrow(0,  size_2, size_3), dim = 1)
        logloss_tgt_4 = torch.log(torch.clamp(x_4 + offset, max=1.))
        classifier_loss_adv_tgt_4 = -F.nll_loss(logloss_tgt_4, target_adv_src_3)
        
        transfer_loss_3 = -(self.srcweight * classifier_loss_adv_src_3 + classifier_loss_adv_tgt_4)

        total_loss_1 = transfer_loss_1+transfer_loss_2+transfer_loss_3

        total_loss_1.backward()
        optimizer1.step()
        
        self.iter_num += 1

    def predict(self, inputs):
        softmax_outputs= self.c_net.inference(inputs)
        return softmax_outputs

    def get_parameter_list(self,flag):
        if flag==1:
            return self.c_net.parameter_list1
        elif flag ==4:
            return self.c_net.parameter_list4      

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode
    
    def get_state(self):
        return self.c_net.state_dict()
