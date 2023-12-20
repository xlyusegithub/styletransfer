import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
# torch.nn.functional.conv2d



class ContentLoss(nn.Module):
    """
    content loss layer
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None
    def forward(self, input):
        self.loss = functional.mse_loss(input, self.target)
        return input
    def update(self, target):
        self.target = target.detach()



class StyleLoss(nn.Module):
    """
    style loss layer
    """
    def __init__(self, target, c_mask, s_mask, patch_size, mrf_style_stride, mrf_synthesis_stride, gpu_chunck_size,
                 device,loc_p=1, glo_p=1, layer_n=0, jia=False,jia_gram=True,out_flag=False):
        super(StyleLoss, self).__init__()
        self.patch_size = patch_size  # 3
        self.mrf_style_stride = mrf_style_stride  # 2
        self.mrf_synthesis_stride = mrf_synthesis_stride  # 2
        self.gpu_chunck_size = gpu_chunck_size  # 256
        self.device = device
        self.gram_part = 1

        assert self.gram_part in [0,1,2, 3], 'gram part is not right,need in [0,1,2, 3]'

        self.loc_param = loc_p
        self.glo_param = glo_p

        self.layer_n = layer_n
        self.jia = jia
        self.jia_gram = jia_gram
        # change_epoch = 0, run_epoch = 0
        self.change_epoch = 0
        self.run_epoch = 0
        self.out_flag=out_flag
        self.single = True
        self.layer_loc = [11, 20]
        self.layer_glob = [3, 8, 15]
        print(self.layer_loc, self.layer_glob, '------------------------------', 147896325)

        self.loss = None
        self.loss_gram = None
        self.loss_gram_part = None
        self.target = target.detach()  #   # 先碎片化，然后特征图连接
        self.s_mask = s_mask.detach()
        self.c_mask = c_mask.detach()

        if self.jia:
            combine_tar = torch.add(target.detach(), s_mask.detach() * self.loc_param)  # 使用加的方式进行处理
        else:
            combine_tar = torch.cat([target.detach(), s_mask.detach() * self.loc_param],
                                    dim=1)
        self.style_patches = self.patches_sampling(combine_tar, patch_size=self.patch_size, stride=self.mrf_style_stride)  # 【batch-new，c，h-patch，w-patch】
        self.style_patches_norm = self.cal_patches_norm(self.style_patches)  # size--【bn， 】
        self.style_patches_norm = self.style_patches_norm.view(-1, 1, 1)
        self.target_gram = self.gram_jisuan(target.detach())  # .detach()

        if self.single:
            self.style_patches_single = self.patches_sampling(target.detach(), patch_size=self.patch_size,
                                                              stride=self.mrf_style_stride)

    def update(self, target, c_mask, s_mask):
        """
        update target of style loss
        :param target:
        :return:
        """
        self.target = target.detach()
        self.s_mask = s_mask.detach()
        self.c_mask = c_mask.detach()

        if self.jia:
            combine_tar = torch.add(target.detach(), s_mask.detach() * self.loc_param)
        else:
            combine_tar = torch.cat([target.detach(), s_mask.detach() * self.loc_param],
                                    dim=1)

        self.style_patches = self.patches_sampling(combine_tar, patch_size=self.patch_size,
                                                   stride=self.mrf_style_stride)  # 【batch-new，c，h-patch，w-patch】
        self.style_patches_norm = self.cal_patches_norm(self.style_patches)  # size--【bn， 】
        self.style_patches_norm = self.style_patches_norm.view(-1, 1, 1)  #
        self.target_gram = self.gram_jisuan(target.detach())  # .detach()

        if self.single:
            self.style_patches_single = self.patches_sampling(target.detach(), patch_size=self.patch_size,
                                                              stride=self.mrf_style_stride)

    def forward(self, input):

        if self.single:
            synthesis_patches_single = self.patches_sampling(input, patch_size=self.patch_size,
                                                             stride=self.mrf_synthesis_stride)

        if self.layer_n in self.layer_loc:  #   if self.layer_n in [11, 20]
            if self.jia:
                combine_syn = torch.add(input, self.c_mask * self.loc_param)
            else:
                combine_syn = torch.cat([input, self.c_mask * self.loc_param], dim=1)

            synthesis_patches = self.patches_sampling(combine_syn, patch_size=self.patch_size,
                                                      stride=self.mrf_synthesis_stride)  # patch-3，stride-2


            max_response = []
            for i in range(0, self.style_patches.shape[0], self.gpu_chunck_size):
                i_start = i

                i_end = min(i + self.gpu_chunck_size,
                            self.style_patches.shape[0])
                weight = self.style_patches[i_start:i_end, :, :,
                         :]
                response = functional.conv2d(combine_syn, weight,
                                             stride=self.mrf_synthesis_stride)
                max_response.append(response.squeeze(dim=0))
            max_response = torch.cat(max_response,
                                     dim=0)
            max_response = max_response.div(
                self.style_patches_norm)
            # max_response = torch.argmin(max_response, dim=0)
            max_response = torch.argmax(max_response, dim=0)
            max_response = torch.reshape(max_response, (1, -1)).squeeze()

            loss = 0
            for i in range(0, len(max_response),self.gpu_chunck_size):  #  h-f*w-f，  【 0--batch-new 】
                i_start = i
                i_end = min(i + self.gpu_chunck_size, len(max_response))
                tp_ind = tuple(range(i_start, i_end))
                sp_ind = max_response[i_start:i_end]
                if self.single:
                    loss += torch.sum(torch.mean(torch.pow(
                        synthesis_patches_single[tp_ind, :, :, :] - self.style_patches_single[sp_ind, :, :, :], 2),
                                                 dim=[1, 2, 3]))
                else:
                    loss += torch.sum(
                        torch.mean(
                            torch.pow(synthesis_patches[tp_ind, :, :, :] - self.style_patches[sp_ind, :, :, :], 2),
                            dim=[1, 2, 3]))
            self.loss = loss / len(max_response)  # input_patch,self.tar_patch    synthesis_patches, self.style_patches

        if self.layer_n in self.layer_glob:  #  [8, 15]    if self.layer_n in [3, 8, 11, 15]:
            if self.jia_gram:
                syn_join = torch.add(input, self.c_mask * self.glo_param)
                tar_join = torch.add(self.target, self.s_mask * self.glo_param)
            else:
                syn_join = torch.cat([input, self.c_mask * self.glo_param],
                                     dim=1)
                tar_join = torch.cat([self.target, self.s_mask * self.glo_param],
                                     dim=1)
                # syn_join = torch.multiply(input, self.c_mask*self.glo_param)  #
                # tar_join = torch.multiply(self.target, self.s_mask*self.glo_param)
            # print(syn_join.shape)
            syn_gram = self.gram_jisuan(syn_join)
            tar_gram = self.gram_jisuan(tar_join)
            self.loss_gram = functional.mse_loss(tar_gram, syn_gram)
        else:
            self.loss_gram = 0

        loss_p = 0
        if self.layer_n in self.layer_loc:  #   if self.layer_n in [11, 20]
            syn_long = synthesis_patches.shape[0]  #
            # print(syn_long)
            # print(max_response.shape)  # torch.Size([961])
            max_response_long = len(max_response)  #  961
            select_patch = list(range(max_response_long))
            select_patch = max_response[select_patch]  #
            choice = np.linspace(0, syn_long - 1, max_response_long)
            select_zhi = [int(i) for i in choice]  #
            '''syn_gram = self.gram_jisuan_part(synthesis_patches[select_zhi, :, :, :],
                                             gram_part=self.gram_part)  #      
            # tar_gram = self.gram_jisuan_part(self.style_patches[select_patch, :, :, :], gram_part=self.gram_part)
            tar_gram = self.gram_jisuan_part(self.style_patches[select_patch, :, :, :],
                                             gram_part=self.gram_part)  #    
            self.loss_gram_part = functional.mse_loss(tar_gram, syn_gram)'''
            for i in range(0, max_response_long,self.gpu_chunck_size):  #  h-f*w-f， 【 0--batch-new 】
                i_start = i
                i_end = min(i + self.gpu_chunck_size, max_response_long)  #
                tp_ind = tuple(range(i_start, i_end))
                sp_ind = max_response[i_start:i_end]  #
                if self.single:
                    syn_gram = self.gram_jisuan_part(synthesis_patches_single[tp_ind, :, :, :],
                                                     gram_part=self.gram_part)
                    tar_gram = self.gram_jisuan_part(self.style_patches_single[sp_ind, :, :, :],
                                                     gram_part=self.gram_part)
                else:
                    syn_gram = self.gram_jisuan_part(synthesis_patches[tp_ind, :, :, :],
                                                     gram_part=self.gram_part)
                    tar_gram = self.gram_jisuan_part(self.style_patches[sp_ind, :, :, :],
                                                     gram_part=self.gram_part)


                loss_p += functional.mse_loss(tar_gram, syn_gram)
                self.loss_gram_part = loss_p

        return input
    def patches_sampling(self, image, patch_size, stride):
        """
        sampling patches form a image
        :param image:
        :param patch_size:
        :return:
        """
        h, w = image.shape[2:4]
        patches = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patches.append(image[:, :, i:i + patch_size, j:j + patch_size])
        patches = torch.cat(patches, dim=0).to(self.device)
        return patches  # 【batch-new，c，h-patch，w-patch】
    def cal_patches_norm(self, style_patches):
        """
        calculate norm of style image patches
        :return:
        """
        norm_array = torch.zeros(style_patches.shape[0])
        for i in range(style_patches.shape[0]):  # self.style_patches = self.patches_sampling(）--【bn，c，size，size】
            norm_array[i] = torch.pow(torch.sum(torch.pow(style_patches[i], 2)), 0.5)
        return norm_array.to(self.device)
    def gram_jisuan_part(self, image_tensr, gram_part=2):  # image_tensor---[b,c,h,w]    gram_part---[0,1,2]
        b, c, h, w = image_tensr.size()
        if gram_part==0:
            F = image_tensr.view(b, c, h, w)
            G = torch.matmul(F, torch.permute(F, [0, 1, 3, 2]))
            G.div_(h * w)
            # G.div_(c*h * w)
        if gram_part==1:
            F = image_tensr.view(b, c, h * w)
            G = torch.bmm(F, F.transpose(1, 2))
            G.div_(h * w * c)  # In-place version of ~Tensor.div   G.div_(h * w*c)
        if gram_part==2:
            F = image_tensr.view(b, c * h * w)
            G = torch.mm(F, F.transpose(0,1))
            G.div_(c*b*h*w)  # In-place version of ~Tensor.div   G.div_(h * w*c)    G.div_(h * w*c*b)
        if gram_part==3:
            F = image_tensr.view(b, 1, c*h * w)
            G = torch.matmul(F, torch.permute(F, [0, 2, 1]))
            G.div_(h * w*c)
        return G
    def gram_jisuan(self, image_tensr):
        b, c, h, w = image_tensr.size()
        F = image_tensr.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w * c)
        return G




'''class TVLoss(nn.Module):  # 使用过程是放置于最前方，应该是使用原图进行处理。应该是对风格迁移图像进行纹理控制，不需要进行特征提取图，直接使用原图
    def __init__(self,image_target, filter_weight):
        """
        tv loss layer
        """
        super(TVLoss, self).__init__()
        self.loss = None
    def forward(self, input):  # 实例化之后要执行方法forward，才能更新self.loss，访问更新之后的属性才不会获得None，会获得真是的loss
        image = input.squeeze().permute([1, 2, 0])  # 应该是转置处理。先squeeze()降维然后进行转置处理
        r = (image[:, :, 0] + 2.12) / 4.37  # 此处应该进行切片降维处理，形成2维数据。从下方temp处看应该使用unsqueeze形成3个维度
        g = (image[:, :, 1] + 2.04) / 4.46
        b = (image[:, :, 2] + 1.80) / 4.44
        temp = torch.cat([r.unsqueeze(2), g.unsqueeze(2), b.unsqueeze(2)], dim=2)
        gx = torch.cat((temp[1:, :, :], temp[-1, :, :].unsqueeze(0)), dim=0)
        gx = gx - temp

        gy = torch.cat((temp[:, 1:, :], temp[:, -1, :].unsqueeze(1)), dim=1)  # 1：，-1两者得到的数据在最后一层应该是重合的
        gy = gy - temp
        self.loss = torch.mean(torch.pow(gx, 2)) + torch.mean(torch.pow(gy, 2))  # 此处应该是正则化 a*a+b*b 处理
        return input
    def update(self, image_target):
        pass  # 对init添加image_target, filter_weight，以及设置update方法应该可以方便对tvloss进行切换，有利于使用不同的tvloss'''





class Filter_weight(nn.Module):
    def forward(self, filter_weight):
        assert type(filter_weight) == np.ndarray
        weight = np.expand_dims(filter_weight, 0)
        print(weight.shape)  # (1, 3, 3)
        weight = np.expand_dims(weight, 0)  # /255
        weight = weight.astype('float32')
        weight = torch.as_tensor(weight)
        weight = torch.cat([weight, weight, weight], dim=0)  #  3
        return weight
weight = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]])
weight_filter = Filter_weight()(weight).cuda()
# print(weight_filter, weight_filter.shape)  #  torch.Size([3, 1, 3, 3])


class TVLoss(nn.Module):  #
    def __init__(self, image_target, filter_weight,avg=False):  # image_tensor---[b,c,h,w]----float(0-1) filter_weight----[3,1,3,3]---tensor
        super(TVLoss, self).__init__()
        self.avg = avg
# class Lap_loss_dfg(nn.Module):
#     def __init__(self, image_target, filter_weight):  # image_tensor---[b,c,h,w]----float(0-1) filter_weight----[3,1,3,3]---tensor
#         super(Lap_loss_dfg, self).__init__()

        self.target = image_target
        if self.avg:
            self.target = functional.avg_pool2d(input=self.target, kernel_size=2, stride=2)

        k = filter_weight.shape[2]
        pad = int((k - 1) / 2)
        self.pad = pad
        self.weight = filter_weight
        self.target_filter = self.filter_image(self.target, self.weight)
        self.syn_filter = None
        self.loss = None

    def forward(self, syn_image):
        if self.avg:
            self.syn_filter = self.filter_image(functional.avg_pool2d(input=syn_image, kernel_size=2, stride=2),
                                                self.weight)
        else:
            self.syn_filter = self.filter_image(syn_image, self.weight)

        self.loss = functional.mse_loss(self.syn_filter, self.target_filter)
        return syn_image
    def filter_image(self, image_tensor, weight):
        out = functional.conv2d(input=image_tensor, weight=weight, padding=self.pad,groups=3)
        return out
    def update(self, image_target):
        self.target = image_target
        if self.avg:
            self.target = functional.avg_pool2d(input=self.target, kernel_size=2, stride=2)
        self.target_filter = self.filter_image(self.target, self.weight)


weight_filter_lap = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]], dtype=np.float32)

class Lap_loss_rgb(nn.Module):
    def __init__(self, image_target, filter_weight,rgb=True, device='cuda', avg=False):  # weight_input, image_tensor---[b,c,h,w]----float(0-1) filter_weight----[3,3]---tensor
        super(Lap_loss_rgb, self).__init__()
        self.device = device
        self.rgb = rgb
        self.avg = avg
        if self.rgb:
            self.repeat = 3
            self.target = image_target[:,0:3, :, :]
            if self.avg:
                self.target = functional.avg_pool2d(input=self.target, kernel_size=2, stride=2)
                pass
            # self.weight = weight_input
        else:
            self.repeat = image_target.shape[1]
            self.target = image_target
            if self.avg:
                self.target = functional.avg_pool2d(input=self.target, kernel_size=2, stride=2)  # 使用池化之后的数据进行处理，
                pass

        k = filter_weight.shape[1]
        pad = int((k - 1) / 2)
        self.pad = pad
        self.weight_in = filter_weight # size------[3,3]
        self.weight = self.filter_weight(self.weight_in, repeat=self.repeat)  # [3,1,3,3]
        self.target_filter = self.filter_image(self.target, self.weight, self.repeat)
        self.syn_filter = None
        self.loss = None
        # print(self.weight, self.weight.shape, self.target_filter.shape)

    def forward(self, syn_image):
        if self.avg:
            self.syn_filter = self.filter_image(functional.avg_pool2d(input=syn_image, kernel_size=2, stride=2),
                                                self.weight, self.repeat)  #
        else:
            self.syn_filter = self.filter_image(syn_image, self.weight, self.repeat)

        self.loss = functional.mse_loss(self.syn_filter, self.target_filter)
        return syn_image
    def filter_image(self, image_tensor, weight, repeat):
        if self.rgb:
            image_tensor_sel = image_tensor[:, 0:3, :, :]
        else:
            image_tensor_sel = image_tensor
        out = functional.conv2d(input=image_tensor_sel, weight=weight, padding=self.pad,groups=repeat)
        return out
    def filter_weight(self, weight_in, repeat):
        assert type(weight_in) == np.ndarray
        weight = weight_in.astype('float32')  #
        weight = torch.from_numpy(weight).to(self.device)  # functional.conv2d same device
        weight = weight.repeat(repeat, 1, 1, 1)  #
        return weight
    def update(self, image_target):
        if self.rgb:
            self.repeat = 3
            self.target = image_target[:, 0:3, :, :]
            if self.avg:
                self.target = functional.avg_pool2d(input=self.target, kernel_size=2, stride=2)
                pass
        else:
            self.repeat = image_target.shape[1]
            self.target = image_target
            if self.avg:
                self.target = functional.avg_pool2d(input=self.target, kernel_size=2, stride=2)
                pass
        self.target_filter = self.filter_image(self.target, self.weight, self.repeat)
