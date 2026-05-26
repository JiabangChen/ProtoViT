# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# code borrowed from: https://github.com/facebookresearch/deit
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from settings import dropout_rate

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

def get_pretrained_weights_path(model_name):

    if model_name in ["deit_small_patch16_224", "deit_base_patch16_224", "deit_tiny_patch16_224",
            "deit_tiny_distilled_patch16_224"]:
        if model_name == "deit_small_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
        elif model_name == "deit_base_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
        elif model_name == "deit_tiny_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'

    return finetune

def get_pretrained_weights(model_name, model):
    finetune = get_pretrained_weights_path(model_name) # 模型预训练权重字典的link
    print(finetune)
    if finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            finetune, map_location='cpu', check_hash=True) # 根据link加载模型的checkpoint，checkpoint中常常包括
        # 模型的参数字典，训练时所用的优化器，lr_scheduler等
    else:
        checkpoint = torch.load(finetune, map_location='cpu')
    checkpoint_model = checkpoint['model'] # 所以要用checkpoint['model']来获取最终的模型的参数字典
    state_dict = model.state_dict() # 这里是我所用timm建立的模型的参数字典
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    # 把官方的预训练权重参数字典中，对于分类头的部分给删掉
    # interpolate position embedding，这里其实是为了提高通用性，即可以容忍不同分辨率的图像的输入，但如果我这边输入图像就是224大小，那便没关系
    # 值得注意的是，对于ViT而言，要想使得能接受不同分辨率的图像，只要改这个position embedding就好了。其他的比如初始patch embedding用到的
    # 卷积，后面生成QKV的线性矩阵，包括MLP和最终分类时用到的linear层，都是和channel数有关，和patch数其实没有关系
    pos_embed_checkpoint = checkpoint_model['pos_embed'] # 官方预训练权重参数字典里的position embedding参数
    embedding_size = pos_embed_checkpoint.shape[-1] # 维度数，vit small是384，tiny是192
    num_patches = model.patch_embed.num_patches # 我所建立的model的patch数，一般是14 x 14 = 196
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 有几个除了正常patch之外的token，比如CLS token和蒸馏token
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5) # 官方预训练权重字典里的patch token的长与宽，比如14 x 14中的14
    # 这里其实是patch token的数量开方，为了是后续做插值，因为插值需要的不是（N, C），而是（C, height, width）
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5) # 我所建立的模型的patch token的长与宽，比如14 x 14中的14
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens] # 官方预训练的extra token的positional embedding,这是保留不变的
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] # 官方预训练的patch token的positional embedding,这是要做插值的
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    # 形状从[1, 196, 384]变成[1, 384, 14, 14]
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    # 做插值，把官方的patch token的positional embedding形成的矩阵向上插值变成和我所建立的模型的patch token的positional embedding
    # 形成的矩阵一样大
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2) # 形状从（1, 384, new_size, new_size）变回（1，new_size * new_size，384）
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    # 把extra token的positional embedding和插值后的patch token的positional embedding拼接起来，形成新的positional embedding矩阵
    checkpoint_model['pos_embed'] = new_pos_embed # 替换

    model.load_state_dict(checkpoint_model, strict=False) # 加载预训练权重的参数字典到我所创建的模型中

    return model



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):

        # not sure if we would employ a teacher model in 
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x#, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch_features(pretrained=False, **kwargs):
    base_arch = 'deit_tiny_patch16_224'
    model = create_model(
    base_arch,
    pretrained=False,
    num_classes=200,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = get_pretrained_weights(base_arch, model)
        del model.head
    return model

@register_model
def deit_small_patch_features(pretrained=False, **kwargs):
    base_arch = 'deit_small_patch16_224'
    model = create_model(
    base_arch,
    pretrained=False,
    num_classes=200,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
    ) # 这里直接调用的是timm中的create_model这个函数，可以根据base_arch直接创建出deit_small_patch16_224这个模型，pretrained设置为False
    # 即在创建模型时，不让timm自行加载预训练权重，后面自行再加载。这样就不需要自己从零搭建deiT的模型了，直接用timm的工具包搭建模型结构就好了
    # 而且创造出来的模型的分类头会按需设置成有200个分类结果的分类头，且其他要求，如drop_rate等也会做好适配。
    model.default_cfg = _cfg() # cfg=configuration，这里是给这个timm模型手动挂上一个“默认配置字典”。通常包括一些metadata信息，
    # 不是模型的可训练参数，不影响 forward 的数学计算，比如只是显示这个模型通常需要什么size的input，如何对输入图片做normalisation等，
    # 但不会真的参与计算，只是一个配置介绍。
    if pretrained:
        model = get_pretrained_weights(base_arch, model) # 给我用timm生成的模型加载预训练权重，但不管分类头的权重
        del model.head # deiT模型作为P方法的backbone不需要分类头，这里直接删掉
    return model