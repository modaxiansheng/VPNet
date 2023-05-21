import torch
from torch.nn import Sequential as Seq
from torch.nn import ModuleList
from .tools import *
from .conv import EdgeConv

class PlainGCNcopy(torch.nn.Module):
    """
    out shape like [input , num_filters[-1]]
    """
    def __init__(self, model_cfg,NUM_FILTERS=[16,32,32], input_channels=16):
        super(PlainGCNcopy, self).__init__()
        self.num_filters = NUM_FILTERS
        self.k = model_cfg.K
        self.act = model_cfg.ACT
        self.norm = model_cfg.NORM
        self.bias = model_cfg.BIAS
        self.dgn = model_cfg.DYN_GRAPH
        self.sum = model_cfg.SUM
        self.fdfs = model_cfg.FDFS
        self.sym = model_cfg.SYMMETRY
        self.merge = model_cfg.MERGE
        self.att = model_cfg.ATT

        if self.sym: #false
            input_channels = int(input_channels/2)

        channels = [input_channels]
        for f_num in self.num_filters:  #num_filters : [64,64,64]
            channels += [f_num] if not self.sym else [int(f_num/2)]
        self.channels = channels #  [4,64,64,64]
        self.num_point_features = channels[-1] if not self.sym else channels[-1]*2  #64
        model_list = []
        for i in range(len(channels) - 1): #012
            in_c = channels[i]
            out_c = channels[i + 1]
            model_list += [EdgeConv(in_c, out_c, act=self.act, norm=self.norm, bias=self.bias, diss=self.fdfs, merge=self.merge, att=self.att)]

        self.models = ModuleList(model_list)

    def forward(self, batch_dict):

        # pointrcnn
        coords = batch_dict['point_coords']
        features = batch_dict['point_features'].unsqueeze(-1)
        # # # ----------------


        # pvrcnn /  second
        # coords = batch_dict.indices   # torch.Size([52724, 4])
        # features = batch_dict.features.unsqueeze(-1)  
        # # ----------------


        #pillar_features : torch.Size([52724, 64])
        #features : torch.Size([52724, 64, 1]
 
        pos = coords[:, 1:4].unsqueeze(-1)  #torch.Size([49081, 3, 1])
        batch_idx = coords[:, 0].long()   #torch.Size([49081])
        
        # print(batch_dict)
        # print('---')
        # print(pos.shape)

        # index = knn(pos, batch_idx, k=self.k,frame_id=batch_dict['frame_id'])
        index = knn1(pos, batch_idx, k=self.k)

        # print('gcn yes')

        for model in self.models:
            if self.dgn:
                index = knn1(features, batch_idx, k=self.kz)
            features = model(features, index, pos)  # torch.Size([47417, 64, 1])

        features = features.squeeze()  #torch.Size([47417, 64])

        # #pointrcnngcn
        if self.sum:
            batch_dict['point_features'] = batch_dict['point_features']+ features
        else:
            batch_dict['point_features'] = features
        # # ----------------

        
        # pvrcnn /  second
        # if self.sum:
        #     batch_dict.features = batch_dict.features+ features
        # else:
        #     batch_dict.features = features


        return batch_dict

    def print(self, batch_dict):
        print('++++++++++++++++++++++++++++++++++++++++')
        print(batch_dict.keys())
        for k,v in batch_dict.items():
            try:
                shape = v.shape
                if len(shape) <= 2:
                    print(k, v.shape)
                    print(v)
                else:
                    print(k, v.shape)

            except:
                print(k, v)
        print('----------------------------------------')