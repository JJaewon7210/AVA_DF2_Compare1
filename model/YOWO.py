import sys
import os
sys.path.append('C:/Users/jw/Documents/CNN/AVA_DF2_Compare1/')


import torch
import torch.nn as nn
import numpy as np
import timm
import math
import yaml
from utils.general import ConfigObject

from model.resnext import resnext101
from model.darknet import Darknet
from model.cfam import CFAMBlock

# Conv3 Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        inter_channels = 1024
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):

        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output
    
# Detection Head
class Detect(nn.Module):
    
    def __init__(self, no=80, anchors=(), ch=(), bbox_head=False):  # detection layer
        super(Detect, self).__init__()
        self.bbox_head = bbox_head # adjust the x, y, w, h for the first 4 channel of detection output
        self.no = no  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        stride = {3: [8, 16, 32], 2: [16, 32], 1: [32]}.get(self.nl, "Unsupported value for self.nl. It must be 1, 2, or 3.")
        self.stride = torch.tensor(stride).float()
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.anchors = a / self.stride.view(-1, 1, 1)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # inference
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            if self.bbox_head:
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        out = (torch.cat(z, 1), x)

        return out
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)

class YOWO(nn.Module):

    def __init__(self, cfg):
        super(YOWO, self).__init__()
        self.cfg = cfg
        
        ##### 2D Backbone #####
        self.backbone_2d = Darknet("cfg/yolo.cfg").cuda()
        num_ch_2d = 425 # Number of output channels for backbone_2d
        if cfg.WEIGHTS.BACKBONE_2D:# load pretrained weights on COCO dataset
            self.backbone_2d.load_weights(cfg.WEIGHTS.BACKBONE_2D)
        if cfg.WEIGHTS.FREEZE_BACKBONE_2D: # freeze
            for param in self.backbone_2d.parameters():
                param.requires_grad = False
            
        ##### 3D Backbone #####
        self.backbone_3d = resnext101().cuda()
        num_ch_3d = 2048 # Number of output channels for backbone_3d
        if cfg.WEIGHTS.BACKBONE_3D:# load pretrained weights on Kinetics-600 dataset
            self.backbone_3d = self.backbone_3d.cuda()
            self.backbone_3d = nn.DataParallel(self.backbone_3d, device_ids=None) # Because the pretrained backbone models are saved in Dataparalled mode
            pretrained_3d_backbone = torch.load(cfg.WEIGHTS.BACKBONE_3D)
            backbone_3d_dict = self.backbone_3d.state_dict()
            pretrained_3d_backbone_dict = {k: v for k, v in pretrained_3d_backbone['state_dict'].items() if k in backbone_3d_dict} # 1. filter out unnecessary keys
            backbone_3d_dict.update(pretrained_3d_backbone_dict) # 2. overwrite entries in the existing state dict
            self.backbone_3d.load_state_dict(backbone_3d_dict) # 3. load the new state dict
            self.backbone_3d = self.backbone_3d.module # remove the dataparallel wrapper
        if cfg.WEIGHTS.FREEZE_BACKBONE_3D: # freeze
            for param in self.backbone_3d.parameters():
                param.requires_grad = False
            
        ##### Attention & Final Conv #####
        self.cfam = CFAMBlock(num_ch_2d+num_ch_3d, 1024)
        self.conv_final = nn.Conv2d(1024, 5*(cfg.MODEL.NUM_CLASSES+4+1), kernel_size=1, bias=False)

        self.seen = 0

    def forward(self, input):
        x_3d = input # Input clip
        x_2d = input[:, :, -1, :, :] # Last frame of the clip that is read

        x_2d = self.backbone_2d(x_2d)
        x_3d = self.backbone_3d(x_3d)
        x_3d = torch.squeeze(x_3d, dim=2)

        x = torch.cat((x_3d, x_2d), dim=1)
        x = self.cfam(x)

        out = self.conv_final(x)

        return out

# Multi-Task Action Fusion 3D (Model)
class YOWO_CUSTOM(YOWO):
    def __init__(self, cfg):
        super(YOWO_CUSTOM, self).__init__(cfg)
        
        self.bs = cfg.batch_size # batch size
        self.anchors = cfg.MODEL.ANCHORS[0]
        self.na = len(self.anchors) // 2 # number of anchors
        self.nl = len(cfg.MODEL.ANCHORS) # number of layers
        self.load_pretrain()
        
        self.convBlock1 = ConvBlock(425, 1024) # for feature x2d
        self.convBlock2 = ConvBlock(1024, 1024) # for feature x2d
        self.heads = Detect(no = 4+1+cfg.nc,
                                anchors = cfg.MODEL.ANCHORS,
                                ch = [1024]*len(cfg.MODEL.ANCHORS),
                                bbox_head=True)

    def forward(self, x):
        # input
        x_3d = x # input clip
        x_2d = x[:, :, -1, :, :] # Last frame of the clip that is read
        
        # Backbone
        x_2d = self.backbone_2d(x_2d)
        x_3d, f_3ds = self.backbone_3d(x_3d)
        x_3d = torch.squeeze(x_3d, dim=2)
        
        x = torch.cat((x_3d, x_2d), dim=1)
        x = self.cfam(x)
        
        ## AVA
        out = self.conv_final(x) # bs, 85*na, 7, 7
        out_pred = out.view(x.shape[0], self.na, 5+80, 7, 7).permute(0, 1, 3, 4, 2)
        out_bbox_pred = out_pred[..., :5]
        out_act_pred = out_pred[..., 5:]
        
        out_infer = self.get_region_boxes(out, None, 80, self.anchors, self.na)
        out_bbox_infer = out_infer[..., :5]
        out_act_infer = out_infer[..., 5:]

        ## DF2
        x = self.convBlock1(x_2d) # for feature x2d
        x = self.convBlock2(x) # for feature x2d
        outs = self.heads([x])
        
        # -> pred & infer
        outs_infer, outs_pred = outs[0], outs[1]
        out_clo_infer = outs_infer[..., 5:]
        out_clo_pred = outs_pred[0][..., 5:]
        
        return [out_bbox_infer, [out_bbox_pred]], [out_clo_infer, [out_clo_pred]], [out_act_infer, [out_act_pred]]
        
    def load_pretrain(self, path=None, freeze=False):
        if path:
            pt_YOWO = torch.load(path)
        else:
            pt_YOWO = torch.load(self.cfg.WEIGHTS.YOWO)
            
        self = nn.DataParallel(self, device_ids=None)
        YOWO_dict = self.state_dict()
        pt_YOWO_dict =  {k: v for k, v in pt_YOWO['state_dict'].items() if k in YOWO_dict} # 1. filter out unnecessary keys
        YOWO_dict.update(pt_YOWO_dict)  # 2. overwrite entries in the existing state dict
        self.load_state_dict(YOWO_dict) # 3. load the new state dict
        self = self.module
        
        if freeze:
            self._freeze_modules()
            
        return self

    def _freeze_modules(self):
        # Freeze all modules in YOWO
        if self.cfg.WEIGHTS.FREEZE_YOWO:
        
            for param in self.backbone_2d.parameters():
                param.requires_grad = False

            # Freeze the 3D backbone
            for param in self.backbone_3d.parameters():
                param.requires_grad = False

            # Freeze the cfam block
            for param in self.cfam.parameters():
                param.requires_grad = False

            # Freeze the final conv block
            for param in self.conv_final.parameters():
                param.requires_grad = False
        

    def get_region_boxes(self, output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False, img_size=224):
        # For inference only
        
        anchor_step = len(anchors)//num_anchors
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch = output.size(0)
        assert(output.size(1) == (5+num_classes)*num_anchors)
        h = output.size(2)
        w = output.size(3)

        all_boxes = [] # inference output
        output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)
        
        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        # xs = (torch.sigmoid(output[0]) + grid_x) /w * img_size
        # ys = (torch.sigmoid(output[1]) + grid_y) /h * img_size
        xs = (torch.sigmoid(output[0]) + grid_x) /w
        ys = (torch.sigmoid(output[1]) + grid_y) /h

        anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        # ws = torch.exp(output[2]) * anchor_w
        # hs = torch.exp(output[3]) * anchor_h
        ws = torch.exp(output[2]) * anchor_w / img_size
        hs = torch.exp(output[3]) * anchor_h / img_size

        det_confs = torch.sigmoid(output[4])
        cls_confs1 = torch.nn.Softmax()(torch.autograd.Variable(output[5:5+14].transpose(0,1))).data
        cls_confs2 = torch.sigmoid(torch.autograd.Variable(output[5+14:5+num_classes].transpose(0,1))).data
        cls_confs = torch.cat([cls_confs1, cls_confs2], dim=1)

        xs = xs.view(batch, num_anchors*h*w, 1)
        ys = ys.view(batch, num_anchors*h*w, 1)
        ws = ws.view(batch, num_anchors*h*w, 1)
        hs = hs.view(batch, num_anchors*h*w, 1)
        det_confs = det_confs.view(batch, num_anchors*h*w, 1)
        cls_confs = cls_confs.view(batch, num_anchors*h*w, num_classes)

        output = torch.cat([xs, ys, ws, hs, det_confs, cls_confs], dim=2)
        
        return output.detach()


if __name__ == '__main__':
    
    with open('cfg/deepfashion2.yaml', 'r') as f:
        _dict_df2 = yaml.safe_load(f)
        opt_df2 = ConfigObject(_dict_df2)
    
    with open('cfg/ava.yaml', 'r') as f:
        _dict_ava = yaml.safe_load(f)
        opt_ava = ConfigObject(_dict_ava)
        
    with open('cfg/model.yaml', 'r') as f:
        _dict_model = yaml.safe_load(f)
        opt_model = ConfigObject(_dict_model)
    
    with open('cfg/hyp.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    
    opt = ConfigObject({})
    opt.merge(opt_df2)
    opt.merge(opt_ava)
    opt.merge(opt_model)
        
    v = np.random.uniform(low=0.0, high=1.0, size= (opt.batch_size,3, opt.DATA.NUM_FRAMES, opt.img_size[0], opt.img_size[0]))
    v = torch.Tensor(v).cuda()
    
    model = YOWO_CUSTOM(cfg = opt).cuda()
    model(v)
    # out_bboxs, out_clos, out_acts = model(v)
    # print('## 01. bbox shape info')
    # for i, j in zip(out_bboxs[0], out_bboxs[1]):
    #     print(i.shape, j.shape)
    # print('## 02. clo shape info')
    # for i in out_clos[0]:
    #     print(i.shape)
    # print('## 03. act shape info')
    # for i in out_acts[0]:
    #     print(i.shape)
