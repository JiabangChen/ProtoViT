import os
import shutil
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import push_greedy
import argparse
import re
import numpy as np
import random

from helpers import makedir
import model
#import push 
#import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma

###################################跑之前可以看下Jiabang's alert##################################3

parser = argparse.ArgumentParser()# 创建一个命令行参数解析器对象parser
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
# 这里用来告诉应该如何解析：比如这里的命令行参数是-gpuid，这个命令行接受一个字符串输入（type=str)
# 且默认值是‘0’，且只接受一个字符串，因为nargs=1，而且这个会让args.gpuid变成一个列表，所以后需要加args.gpuid[0]才可以读取
# 这个nargs，type，default就类似于命令行参数的输入类型（输入数量，type，默认值）
args = parser.parse_args() # 从命令行中实际读取用户输入，并将其解析成一个对象 args。args的属性名就是命令的参数名，这里是gpuid，
# 对应的值就是用户在命令行中输入的值形成的列表
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0] # 指定pytorch用哪几块GPU，这里是0，1，2，3共四块GPU
print(os.environ['CUDA_VISIBLE_DEVICES']) # 在跑之前要查一下有几块GPU，一般我只有1块 Jiabang's alert

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
# base_architecture = 'deit_small_patch16_224', 这里输出deit

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size,sig_temp, radius

normalize = transforms.Normalize(mean=mean,
                                 std=std)
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     #random.seed(seed)  # Python random module.
#     torch.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

seed = np.random.randint(10, 10000, size=1)[0]
# 用 NumPy 随机生成一个整数，范围是 [10, 10000)，也就是 10 到 9999。size=1 返回的是长度为 1 的数组，比如 array([4321])，最后 [0] 取出里面那个整数。
torch.manual_seed(seed) # 把这个随机数设置为 PyTorch CPU 端的随机种子
torch.cuda.manual_seed(seed) # 把同一个随机数设置为当前 CUDA GPU 的随机种子
# 这里比较特殊的是：它不是固定 seed，而是每次运行程序时先随机生成一个 seed。也就是说，这段代码不能保证每次实验完全可复现；
# 它只是保证“本次运行内部”CPU 和 GPU 使用同一个 seed。如果想要实验能复现，seed要是一个固定的数字
#set_seed(seed)
# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True) # Jiabang's change,删去了num_workers=4, pin_memory=False, drop_last=True
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False) # Jiabang's change,删去了num_workers=4, pin_memory=False
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False) # Jiabang's change,删去了num_workers=4, pin_memory=False

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture, # deit_small_patch16_224
                              pretrained=True, img_size=img_size, # 224
                              prototype_shape=prototype_shape, # (2000, 384, 4)
                              num_classes=num_classes, # 200
                              prototype_activation_function=prototype_activation_function, # log
                              sig_temp = sig_temp, # 100
                              radius = radius, # 1
                              add_on_layers_type=add_on_layers_type) # regular

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(str(device))
ppnet = ppnet.to(device)
model_ema = None
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size, k,stage_2_lrs

joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 #{'params': ppnet.patch_select, 'lr': joint_optimizer_lrs['patch_select']},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.AdamW(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
# scheduler是stepLR，每过5个epoch，lr*0.1,这个是训练的stage 1，只关注于backbone和prototype
# to train the slots 
joint_optimizer_specs_stage2 =[{'params': ppnet.patch_select, 'lr': stage_2_lrs['patch_select']}]

joint_optimizer2 = torch.optim.AdamW(joint_optimizer_specs_stage2)
joint_lr_scheduler2 = torch.optim.lr_scheduler.StepLR(joint_optimizer2, step_size=joint_lr_step_size, gamma=0.1)
# 同样，scheduler是stepLR，每过5个epoch，lr*0.1,这个是训练的指示函数中的那个vector v

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': warm_optimizer_lrs['features'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.AdamW(warm_optimizer_specs)
# 这里与传统P方法不一样的是，warm-up中是对P参数和backbone参数更新

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start,sum_cls, push_start, slots_train_epoch 

# train the model
log('start training')

log(f'weight coefs are: {coefs}')
import copy

# check_base = False 
# if check_base:
#     # check the performance of base-arch 
#     n_examples = 0
#     n_correct = 0
#     for i, (image, label) in enumerate(test_loader):
#         image = image.to(device)
#         label = label.to(device)
#         out = ppnet.features(image)
#         _, predicted = torch.max(out.data, 1)
#         n_examples += label.size(0)
#         n_correct += (predicted == label).sum().item()
#     log('base-arch acc: \t{0}'.format(n_correct / n_examples * 100))

#slots_epoch = num_warm_epochs +5 
# only_push = False 
# if not only_push:
# not ready for push yet
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs: # 前5个epoch先做warm-up
        tnt.warm_only(model=ppnet, log=log)
        _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=warm_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k, sum_cls = sum_cls)
    else:
        tnt.joint(model=ppnet, log=log)
        # to train the model with no slots 
        _ , train_loss= tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k,sum_cls = sum_cls)
        joint_lr_scheduler.step()

    accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                    class_specific=class_specific, log=log, clst_k=k,sum_cls = sum_cls)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.75, log=log)
    # version 1, learn slots before push 
from settings import coefs_slots
coh_weight = coefs_slots['coh']
coefs['coh']  = coh_weight # 在微调slots时，更新coherence loss的权重
log(f'Coefs for slots training: {coefs}')
for epoch in range(slots_train_epoch):
    tnt.joint(model=ppnet, log=log)
    log('epoch: \t{0}'.format(epoch))
    _ , train_loss= tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer2,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k,sum_cls = sum_cls)
    joint_lr_scheduler2.step()
    accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                    class_specific=class_specific, log=log, clst_k=k, sum_cls = sum_cls)
    
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'slots', accu=accu,
                                target_accu=0.75, log=log)

push_greedy.push_prototypes(
        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
        pnet = ppnet, # pytorch network with prototype_vectors
        class_specific=class_specific,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
        prototype_img_filename_prefix=prototype_img_filename_prefix,
        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
        save_prototype_class_identity=True,
        log=log)
accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                class_specific=class_specific, log=log, clst_k = k,sum_cls = sum_cls)
save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                            target_accu=0.0, log=log)

for epoch in range(15):
    tnt.last_only(model=ppnet, log=log)
    log('iteration: \t{0}'.format(epoch))
    _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer,
                class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k, sum_cls = sum_cls)
    print('Accuracy is:')
    accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                        class_specific=class_specific, log=log, clst_k = k,sum_cls = sum_cls)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'finetuned', accu=accu,
                            target_accu=0.70, log=log)
logclose()

