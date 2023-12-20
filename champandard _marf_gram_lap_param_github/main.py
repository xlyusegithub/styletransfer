from model import CNNMRF
import torch.optim as optim
from torchvision import transforms
import cv2
import argparse
import torch
import torchvision
import torch.nn.functional as functional
from PIL import Image
# torch.nn.functional.interpolate
import time
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
reference:
[1]. Li C, Wand M. Combining markov random fields and convolutional neural networks for image synthesis[C].
//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2479-2486.

[2]. https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/neural_style_transfer/main.py

[3]. https://heartbeat.fritz.ai/neural-style-transfer-with-pytorch-49e7c1fe3bea
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def txt_write(path, outcome, flag='a'):  # path---txt, outcome---list,tuple,【list，tuple，dict】。尽量单一列表保存为txt时不要有
    # 字母，数字混合，以免无法分离。混合列表，字典等可以使用字母，数字混合
    with open(path, flag) as f:
        for i in outcome:
            i = str(i)  # write要使用 字符串  数据进行输入
            f.write(i)  # 如果 i 是字符串数据，可以将 ‘/n’ 写在一起
            f.write('\n')


def unsample_synthesis(height, width, synthesis, device):  # size
    """
    unsample synthesis image to next level of training
    :param height: height of unsampled image
    :param width: width of unsampled image
    :param synthesis: synthesis image tensor to unsample
    :param device:
    :return:
    """
    # transform the tensor to numpy, and upsampled as a image
    synthesis = functional.interpolate(synthesis, size=[height, width], mode='bilinear')
    # finally, set requires grad, the node will be leaf node and require grad
    synthesis = synthesis.clone().detach().requires_grad_(True).to(device)
    # synthesis = synthesis.clone().detach().requires_grad_(True).to(device)
    return synthesis


def load_pre_image_totensor(image_path, img_size=[256, 256], data_range=255):
    prep = transforms.Compose([transforms.Resize(img_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),

                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(data_range))
                               ])
    img = Image.open(image_path).convert("RGB")  # .convert("RGBA")
    return prep(img).unsqueeze(0)  # [b， c， h， w】


def post_image_toTensor(image_tensor, data_range=255):
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. /data_range)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])
                                 ])  #,
    image_t = postpa(image_tensor.detach().clone()[0])
    image_t.clamp_(0, 1)
    # image_t = transforms.ToPILImage()(image_t)
    return image_t


def load_pre_image_totensor369(image_path, img_size=[256, 256], data_range=255):
    prep = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                               transforms.Lambda(lambda x: x.mul_(data_range))
                               ])
    img = Image.open(image_path).convert("RGB")
    return prep(img).unsqueeze(0)  # [b， c， h， w】
def post_image_toTensor369(image_tensor, data_range=255):
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. /data_range)),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])])  #,
    image_t = postpa(image_tensor.detach().clone()[0])
    image_t.clamp_(0, 1)
    return image_t


def save_Tensorimage(image_Tensor, save_path):  # 【c，h，w】
    torchvision.utils.save_image(image_Tensor, save_path)

def jpg_png(img_path):  # 'asd/wert/tyu.jpg'------tyu.png
    sp = os.path.split(img_path)  # ('asd/wert', 'tyu.jpg')
    s1 = os.path.splitext(sp[1])  # ('tyu', '.jpg')
    return s1[0]+'.'+'png'
def path_name_split(img_path):  # 'asd/wert/fig_name.jpg'------fig_name
    sp = os.path.split(img_path)  # ('asd/wert', 'fig_name.jpg')
    s1 = os.path.splitext(sp[1])  # ('fig_name', '.jpg')
    return s1[0]  # fig_name


def main(config):
    plots_loss = []
    save_fig_name = path_name_split(config.content_path) + '_' + path_name_split(config.style_path)
    # print(save_fig_name)  # a123_b369
    save_path = config.synthesis_path + '/' + save_fig_name + "_" + '100'  # synthetic_image/a123_b369
    # print(save_path)  # synthetic_image/a123_b369_100
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    "-------------------transform and denorm transform-----------------"
    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406]

    "--------------read image------------------"
    if not os.path.exists(config.content_path):
        raise ValueError('file %s does not exist.' % config.content_path)
    if not os.path.exists(config.style_path):
        raise ValueError('file %s does not exist.' % config.style_path)


    c_mask_pathjia = config.mask_file + '/' + path_name_split(config.content_path) + '.png'
    s_mask_pathjia = config.mask_file + '/' + path_name_split(config.style_path) + '.png'
    if os.path.exists(c_mask_pathjia) and os.path.exists(s_mask_pathjia):
        pass
    else:
        c_mask_pathjia = config.content_path
        s_mask_pathjia = config.style_path
    size_f = 256  # control size
    if isinstance(size_f, (list, tuple)):
        size_f_h = size_f[0]
        size_f_w = size_f[1]
    else:
        size_f_h = size_f
        size_f_w = size_f
    data_range = 255  # 1,30,60,90,150,255
    c_mask_image = load_pre_image_totensor(c_mask_pathjia, img_size=[size_f_h, size_f_w], data_range=data_range).to(device)  #
    s_mask_image = load_pre_image_totensor(s_mask_pathjia, img_size=[size_f_h, size_f_w], data_range=data_range).to(device)
    content_image = load_pre_image_totensor(config.content_path, img_size=[size_f_h, size_f_w], data_range=data_range).to(device)  #
    style_image = load_pre_image_totensor(config.style_path, img_size=[size_f_h, size_f_w], data_range=data_range).to(device)
    pyramid_content_image = []
    pyramid_style_image = []
    pyramid_cmask_image = []
    pyramid_smask_image = []
    for i in range(config.num_res):  # num_res', type=int, default=3)
        content = functional.interpolate(content_image, scale_factor=1/pow(2, config.num_res-1-i), mode='bilinear')
        style = functional.interpolate(style_image, scale_factor=1/pow(2, config.num_res-1-i), mode='bilinear')
        cmask = functional.interpolate(c_mask_image, scale_factor=1/pow(2, config.num_res - 1 - i), mode='bilinear')
        smask = functional.interpolate(s_mask_image, scale_factor=1 / pow(2, config.num_res - 1 - i), mode='bilinear')

        pyramid_content_image.append(content)
        pyramid_style_image.append(style)
        pyramid_cmask_image.append(cmask)
        pyramid_smask_image.append(smask)

    "-----------------start training-------"

    synthesis = None  #
    synthesis = content_image.clone().detach().requires_grad_(True)
    # create cnnmrf model

    cnnmrf = CNNMRF(style_image=pyramid_style_image[0], content_image=pyramid_content_image[0], c_mask_image=pyramid_cmask_image[0], s_mask_image=pyramid_smask_image[0],device=device,
                    content_weight=config.content_weight, style_weight=config.style_weight,
                    style_gram_weight=config.style_gram_weight, style_gram_weight_part=config.style_gram_weight_part,tv_weight=config.tv_weight,lap_weight=config.lap_weight,
                    gpu_chunck_size=config.gpu_chunck_size, mrf_synthesis_stride=config.mrf_synthesis_stride,patch_size=config.patch_size,
                    mrf_style_stride=config.mrf_style_stride, loc_p=config.loc_p,
                    glo_p=config.glo_p, lap_rgb=config.lap_rgb, jia=config.jia, jia_gram=config.jia_gram,out_flag=config.out_flag).to(device)

    # Sets the module in training mode.

    cnnmrf.train()
    for i in range(0, config.num_res):
        if i == 0:
            # in lowest level init the synthesis from content resized image
            synthesis = pyramid_content_image[0].clone().requires_grad_(True).to(
                device)  # 直接设置.requires_grad_(True)
        else:
            synthesis = unsample_synthesis(pyramid_content_image[i].shape[2], pyramid_content_image[i].shape[3],
                                           synthesis, device)
            cnnmrf.update_style_and_content_image(style_image=pyramid_style_image[i],
                                                  content_image=pyramid_content_image[i],
                                                  c_mask=pyramid_cmask_image[i], s_mask=pyramid_smask_image[i])

        optimizer = optim.LBFGS([synthesis], lr=1, max_iter=1)
        iter = [0]
        count = config.max_iter
        while iter[0] < count:
            def closure():
                optimizer.zero_grad()
                loss = cnnmrf(synthesis,change_epoch=i,run_epoch=iter[0])
                loss.backward()
                if (iter[0] + 1) % 5 == 0:
                    plots_loss.append(loss.item())
                # save image
                if (iter[0] + 1) % config.sample_step == 0 or iter[0] + 1 == config.max_iter:
                    print('res-%d-iteration-%d: %f' % (i, iter[0] + 1, loss.item()))
                    print('save image: res-%d-result-%d.jpg' % (i, iter[0] + 1))
                    image = post_image_toTensor(synthesis, data_range=data_range)
                    image = functional.interpolate(image.unsqueeze(0), size=content_image.shape[2:4], mode='bilinear')
                    path = f"epoch{i}_{save_fig_name}_{iter[0] + 1}.jpg"  # epoch0_a123_b369_50.jpg
                    path = save_path + '/' + path  # synthetic_image/xinjian1/epoch0_a123_b369_50.jpg
                    save_Tensorimage(image[0], path)
                iter[0] = iter[0]+1
                return loss
            optimizer.step(closure)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 以下数据进行本次使用运行

    # parser.add_argument('--content_path', type=str, default='con_sty_image/in14.png')
    # parser.add_argument('--style_path', type=str, default='con_sty_image/tar14.png')

    # parser.add_argument('--content_path', type=str, default='con_sty_image/in45.png')
    # parser.add_argument('--style_path', type=str, default='con_sty_image/tar45.png')

    # parser.add_argument('--content_path', type=str, default='con_sty_image/tu16.jpg')  # tu14.tu16.
    # parser.add_argument('--style_path', type=str, default='con_sty_image/in16.png')  # in16，tar49，tar16，57png，mosaic。jpg

    # parser.add_argument('--content_path', type=str, default='con_sty_image/tu16.jpg')
    # parser.add_argument('--style_path', type=str, default='con_sty_image/mosaic.jpg')

    parser.add_argument('--content_path', type=str, default='con_sty_image/tu15.jpg')
    parser.add_argument('--style_path', type=str, default='con_sty_image/tiger.jpg')  # mosaic.jpg    tiger.jpg  tar13.png

    parser.add_argument('--c_mask_path', type=str, default='./data/a123.jpg')
    parser.add_argument('--s_mask_path', type=str, default='./data/b369.jpg')
    parser.add_argument('--mask_file', type=str, default='con_sty_mask')
    parser.add_argument('--synthesis_file', type=str, default='xinjian1')
    parser.add_argument('--synthesis_path', type=str, default='synthetic_image')
    parser.add_argument('--content_weight', type=float, default=5)  # 1.75
    parser.add_argument('--style_weight', type=float, default=0.5)
    parser.add_argument('--style_gram_weight', type=float, default=0.2)  # 0.3
    parser.add_argument('--style_gram_weight_part', type=float, default=0.1)
    parser.add_argument('--tv_weight', type=float, default=30e-3)  #
    parser.add_argument('--lap_weight', type=float, default=10e-3)  #
    parser.add_argument('--loc_p', type=float, default=1)
    parser.add_argument('--glo_p', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--mrf_style_stride', type=int, default=2)
    parser.add_argument('--mrf_synthesis_stride', type=int, default=2)
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--num_res', type=int, default=3)
    parser.add_argument('--lap_rgb', type=bool, default=True)
    parser.add_argument('--jia', type=bool, default=False)
    parser.add_argument('--jia_gram', type=bool, default=True)
    parser.add_argument('--out_flag', type=bool, default=True)  # out_flag = True
    parser.add_argument('--gpu_chunck_size', type=int, default=64)  #

    config = parser.parse_args()
    print(config)
    main(config)
