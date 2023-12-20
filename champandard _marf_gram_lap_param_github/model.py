from __future__ import print_function
import torch.nn as nn
import torchvision.models as models
from mylibs import ContentLoss, StyleLoss, TVLoss, Lap_loss_rgb
import numpy as np
import torch
import torch.nn.functional as functional





vgg19 = models.vgg19(weights='VGG19_Weights.DEFAULT')  # `weights=VGG19_Weights.IMAGENET1K_V1`. You can also use `weights=VGG19_Weights.DEFAULT`
for i, v in enumerate(vgg19.features):#children():
    # print(i, v)
    pass
vgg19_long = len(vgg19.features)
# print(vgg19_long)  # 37
n_mod123 = list(vgg19.features[0:vgg19_long-1])
# print(len(n_mod123))  # 36
# print(n_mod123)
def external_layers(pooling='avg', poolsize=2):
    ex_layer = []
    for i in range(1, 4):
        if pooling=='avg':
            layer = nn.AvgPool2d(kernel_size=2**i, stride=2**i)
            ex_layer.append(layer)
            pass
        if pooling=='max':
            layer = nn.MaxPool2d(kernel_size=2**i, stride=2**i)
            ex_layer.append(layer)
            pass
    return ex_layer
layers = external_layers()
print(layers)
for i in layers:
    # print(i)
    pass

full = n_mod123 + layers
print(type(full))  # list.
for i in full:
    # print(i)
    pass
print(len(full))  # 39
vgg19_models_new = nn.Sequential(*full).to('cuda')  # 不使用后方会报错Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
print(vgg19_models_new[33])
for param in vgg19_models_new.parameters():  # 后续添加程序
    param.requires_grad = False

# a = 7
# assert  a< 0, "qwert"


class Filter_weight(nn.Module):
    def forward(self, filter_weight):
        assert type(filter_weight) == np.ndarray
        weight = np.expand_dims(filter_weight, 0)
        print(weight.shape)  # (1, 3, 3)
        weight = np.expand_dims(weight, 0)
        weight = weight.astype('float32')
        weight = torch.as_tensor(weight)
        weight = torch.cat([weight, weight, weight], dim=0)  # 3
        return weight
weight = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
weight_filter = Filter_weight()(weight)

weight_filter_lap = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]])
weight_filter_lap = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]], dtype=np.float32)


class CNNMRF(nn.Module):
    def __init__(self, style_image, content_image, c_mask_image, s_mask_image,device, content_weight=1,lap_weight=0.003,
                 style_weight=0.4, style_gram_weight=0.4,style_gram_weight_part=0.4, tv_weight=0.1,
                 gpu_chunck_size=256, mrf_style_stride=2,patch_size=3,
                 mrf_synthesis_stride=2, weight = weight_filter, loc_p=1, glo_p=1, lap_rgb=True, jia=False,jia_gram=True,
                 out_flag = False):
        super(CNNMRF, self).__init__()
        # fine tune alpha_content to interpolate between the content and the style
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.style_gram_weight = style_gram_weight
        self.gram_part = 1  # [0,1,2] gram计算方式的3种选择
        self.style_gram_weight_part = style_gram_weight_part
        self.tv_weight = tv_weight
        self.lap_weight = lap_weight
        self.patch_size = patch_size
        self.device = device
        self.rgb = lap_rgb
        self.jia = jia
        self.jia_gram = jia_gram
        self.out_flag = out_flag
        self.c_layer = [22]
        self.s_layer = [3,8,11,15,20]
        self.s_loc = [2, 4]
        self.s_glob = [0,1,3]
        assert min(self.s_loc) >=0 and max(self.s_loc)<len(self.s_layer), 'input data is wrong'
        assert min(self.s_glob) >= 0 and max(self.s_glob) < len(self.s_layer), 'input data is wrong'

        self.gpu_chunck_size = gpu_chunck_size  # 256
        self.mrf_style_stride = mrf_style_stride
        self.mrf_synthesis_stride = mrf_synthesis_stride
        self.loc_p = loc_p
        self.glo_p = glo_p
        self.filter_weight = weight.to(self.device)

# 1-1，1-2，2-1，2-2，3-1，3-2，3-3，4-1 ----------1, 3，6,8， 11,13，15， 20,     3-1，4-1 局部   1-2，2-2，3-3使用gram   3，8， 11,15， 20,   3，8，15--11，20
#         self.style_layers = [3,8,11,15,20] # 'p3','r53'      [1, 6, 11, 20,29][22]  [2, 7, 12, 21][23]--relu  [0, 5, 10, 19]------会报错
#         self.content_layers = [22]  # [22]  4-2


        self.lap_layers = [9]
        self.style_layers = self.s_layer
        self.content_layers = self.c_layer

        self.model, self.content_losses, self.style_losses, self.tv_losses , self.lap_losses= \
            self.get_model_and_losses(style_image=style_image, content_image=content_image, c_mask=c_mask_image,
                                      s_mask=s_mask_image)

    def forward(self, synthesis,change_epoch,run_epoch):
        self.model(synthesis)

        # print(len(self.style_losses), '---------------------------------long')

        sty_gram_score = 0
        sty_gram_part_score = 0
        style_score = 0
        content_score = 0
        lap_score = 0

        tv_score = self.tv_losses[0].loss
        for i,sl in enumerate(self.style_losses):
            if i in self.s_loc:  # if i in [2, 4]
                # style_score += sl.loss
                style_score = sl.loss + style_score
            if i in self.s_loc:  # if i in [2, 4]
                # sty_gram_part_score += sl.loss_gram_part
                sty_gram_part_score = sl.loss_gram_part + sty_gram_part_score
            if i in self.s_glob:  # if i in [1,3]   01, 3
                # sty_gram_score += sl.loss_gram
                sty_gram_score = sl.loss_gram + sty_gram_score
        for cl in self.content_losses:
            # content_score += cl.loss
            content_score = cl.loss + content_score
        for lap_l in self.lap_losses:
            # lap_score += lap_l.loss
            lap_score = lap_l.loss + lap_score
        loss = self.content_weight * content_score + self.style_weight * style_score + self.style_gram_weight * sty_gram_score + \
               self.style_gram_weight_part*sty_gram_part_score +  \
               self.tv_weight * tv_score + self.lap_weight*lap_score
        return loss  #

    def update_style_and_content_image(self, style_image, content_image, c_mask, s_mask):
        """
        update the target of style loss layer and content loss layer
        :param style_image:
        :param content_image:
        :return:
        """

        self.tv_loss = self.tv_losses[0].update(content_image.clone())
        x = style_image.clone()
        c_img = c_mask.clone()
        s_img = s_mask.clone()
        next_style_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TVLoss) or isinstance(layer, ContentLoss) or isinstance(layer, StyleLoss) or isinstance(layer, Lap_loss_rgb):
                continue
            if next_style_idx >= len(self.style_losses):
                break
            x = layer(x)  #
            c_img = layer(c_img)  # c_img = layer(c_img)   nn.ReLU().
            s_img = layer(s_img)
            if i in self.style_layers:
                self.style_losses[next_style_idx].update(x, c_img, s_img)
                next_style_idx += 1
            i += 1

        # update the target of content loss layer
        xlap = content_image.clone()
        next_content_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TVLoss) or isinstance(layer, ContentLoss) or isinstance(layer, StyleLoss) or isinstance(layer, Lap_loss_rgb):
                continue
            if next_content_idx >= len(self.content_losses):
                break
            xlap = layer(xlap)
            if i in self.content_layers:  # self.content_layers = [22]
                # extract feature of content image in vgg19 as content loss target
                # xc = functional.relu(xc)
                self.content_losses[next_content_idx].update(xlap)
                next_content_idx += 1
            i += 1
            # update the target of lap loss layer
        xc = content_image.clone()
        next_lap_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TVLoss) or isinstance(layer, ContentLoss) or isinstance(layer,StyleLoss) or isinstance(layer, Lap_loss_rgb):
                continue
            if next_lap_idx >= len(self.lap_losses):
                break
            xc = layer(xc)  # layer
            if i in self.lap_layers:  # self.lap_layers = [9]
                # extract feature of content image in vgg19 as content loss target
                # xc = functional.relu(xc)
                self.lap_losses[next_lap_idx].update(xc)
                next_lap_idx += 1
            i += 1

    def get_model_and_losses(self, style_image, content_image, c_mask, s_mask):

        # vgg = models.vgg19(pretrained=True).to(self.device)  #
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(self.device)  #
        for param in vgg.parameters():
            param.requires_grad = False
        model = nn.Sequential()
        content_losses = []
        style_losses = []
        tv_losses = []
        lap_losses = []  # Lap_loss_rgb
        # add tv loss layer
        tv_loss = TVLoss(content_image, self.filter_weight)  #
        model.add_module('tv_loss', tv_loss)
        tv_losses.append(tv_loss)  #

        next_content_idx = 0
        next_style_idx = 0
        next_lap_idx = 0
        # for i in range(len(vgg19_models_new)):  # layer = vgg19_models_new[i]
        for i in range(len(vgg.features)):  # vgg = models.vgg19(pretrained=True).to(self.device)

            if next_content_idx >= len(self.content_layers) and next_style_idx >= len(self.style_layers) and next_lap_idx >= len(self.lap_layers):
                break
            # add layer of vgg19
            layer = vgg.features[i]
            # layer = vgg19_models_new[i]  #    for i in range(len(vgg19_models_new)):

            # if type(layer)==nn.MaxPool2d:
            #     layer =nn.AvgPool2d(kernel_size=2, stride=2)
            if type(layer)==nn.ReLU:
                layer = nn.ReLU(inplace=False)

            name = str(i)  # 模型中用于模型层名的使用
            model.add_module(name, layer)


            if i in self.content_layers:  # self.content_layers=【22】。

                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(next_content_idx), content_loss)
                content_losses.append(content_loss)
                next_content_idx += 1

            # add style loss layer
            if i in self.style_layers:  # self.style_layers = [11, 20]。
                target_feature = model(style_image).detach()
                c_mask_feature = model(c_mask).detach()
                s_mask_feature = model(s_mask).detach()
                style_loss = StyleLoss(target_feature, c_mask=c_mask_feature, s_mask=s_mask_feature, patch_size=self.patch_size, mrf_style_stride=self.mrf_style_stride,
                                       mrf_synthesis_stride=self.mrf_synthesis_stride, gpu_chunck_size=self.gpu_chunck_size,
                                       device=self.device,loc_p=self.loc_p, glo_p=self.glo_p,
                                       jia=self.jia, jia_gram=self.jia_gram,layer_n=i)
                model.add_module("style_loss_{}".format(next_style_idx), style_loss)  # self.loc_p = loc_p   self.glo_p = glo_p
                style_losses.append(style_loss)
                next_style_idx += 1
            if i in self.lap_layers:

                target_lap = model(content_image).detach()
                lap_loss = Lap_loss_rgb(target_lap, filter_weight=weight_filter_lap,rgb=self.rgb, device=self.device)  # device='cuda'
                model.add_module("lap_loss_{}".format(next_lap_idx), lap_loss)
                lap_losses.append(lap_loss)
                next_lap_idx += 1
        return model, content_losses, style_losses, tv_losses, lap_losses
