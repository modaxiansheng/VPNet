import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    B, _, N = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

    return idx, pairwise_distance

def get_scorenet_input_2(pos, idx, k, batch_idx):
    # pos n 3 1  
    # idx n k
    """(neighbor, neighbor-center)"""
    # batch_size = pos.size(0)

    batch_size = torch.max(batch_idx) + 1
    num_points = pos.size(0)
    _, num_dims, _ = pos.size()
    # pos = pos.view(batch_size, -1, num_points) #b c n  # 2 3 6935

    device = torch.device('cuda')
    

    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points  #n 1

    # idx = idx + idx_base   # n k

    idx = idx.view(-1)  #nk


    # pos = pos.transpose(2, 1).contiguous() # b n c
    
    neighbor = pos.squeeze(-1)[idx, :]  # nk c

    neighbor = neighbor.view(1,num_points, k, num_dims) #1 n k c


    pos = pos.view(1, num_points, 1, num_dims).repeat(1,1, k, 1) #1 n k c

    xyz = torch.cat((neighbor - pos, neighbor), dim=3).permute(0, 3, 1, 2).contiguous() # b, 6,n,k    6 = 3+3
    # n k 2c   --->  1 6 n k
    return xyz

def get_scorenet_input_1(pos, idx, k, batch_idx):
    # pos n 3 1  
    # idx n k
    """(neighbor, neighbor-center)"""
    # batch_size = pos.size(0)

    batch_size = torch.max(batch_idx) + 1
    num_points = pos.size(0)
    _, num_dims, _ = pos.size()
    # pos = pos.view(batch_size, -1, num_points) #b c n  # 2 3 6935

    device = torch.device('cuda')
    

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points  #n 1

    idx = idx + idx_base   # n k

    idx = idx.view(-1)  #nk


    # pos = pos.transpose(2, 1).contiguous() # b n c
    
    neighbor = pos.squeeze(-1)[idx, :]  # nk c

    neighbor = neighbor.view(num_points, k, num_dims) # n k c


    pos = pos.view(num_points, 1, num_dims).repeat(1, k, 1) # n k c

    xyz = torch.cat((neighbor - pos, neighbor), dim=2).permute(2,0,1).unsqueeze(0) # b, 6,n,k    6 = 3+3
    # n k 2c   --->  1 6 n k
    return xyz

def get_scorenet_input(x, idx, k):
    # x is pos (xyz)
    """(neighbor, neighbor-center)"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points) #b c n  # 2 3 6935

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  
    # idx       torch.Size([2, 6624, 20])  
    # idx_base  torch.Size([2, 1, 1])
    idx = idx + idx_base

    idx = idx.view(-1)  #torch.Size([277400])  #bnk

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() # b n c

    #neighbor  邻居的xyz坐标
    neighbor = x.view(batch_size * num_points, -1)[idx, :] #(bn) c  [bnk,:]    #    #torch.Size([277400, 3]) bnk c

    neighbor = neighbor.view(batch_size, num_points, k, num_dims) # b n k c  

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # b n k c

    xyz = torch.cat((neighbor - x, neighbor), dim=3).permute(0, 3, 1, 2).contiguous()  # b,6,n,k    6 = 3+3
    # 邻居的坐标- 中心坐标,   邻居坐标
    return xyz


def feat_trans_dgcnn(point_input, kernel, m):
    '''
    point_input: b   c     n
    kernel:      2i1 mi1o1
    '''

    """transforming features using weight matrices"""
    # following get_graph_feature in DGCNN: torch.cat((neighbor - center, neighbor), dim=3)
    B, _, N = point_input.size()  # b, c, n
    
    point_output = torch.matmul(point_input.permute(0, 2, 1).repeat(1, 1, 2), kernel).view(B, N, m, -1)
      #b n c -> b n 2c @ 2i mi1o1--> b n mo    # b,n,m,o
    center_output = torch.matmul(point_input.permute(0, 2, 1), kernel[:point_input.size(1)]).view(B, N, m, -1)  # b,n,m,cout
    return point_output, center_output


def feat_trans_pointnet(point_input, kernel, m):
    """transforming features using weight matrices"""
    # no feature concat, following PointNet
    B, _, N = point_input.size()  # b, cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1), kernel).view(B, N, m, -1)  # b,n,m,cout
    return point_output


class ScoreNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[16], last_bn=False,norm=False):  #last_bn 最后一层加bn, norm之前的层是否加bn
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.bnorm = norm

        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs_nohidden = nn.Conv2d(in_channel, out_channel, 1, bias=not last_bn)
            if self.last_bn:
                self.mlp_bns_nohidden = nn.BatchNorm2d(out_channel)

        else:
            self.mlp_convs_hidden.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))  # from in_channel to first hidden
            if self.bnorm:
                self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):  # from 2nd hidden to next hidden to last hidden
                self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                if self.bnorm:
                    self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=not last_bn))  # from last hidden to out_channel
            if self.bnorm:
                self.mlp_bns_hidden.append(nn.BatchNorm2d(out_channel))

    def forward(self, xyz, calc_scores='softmax', bias=0):
        ''' input:  B 2C N K
            output: B N  K m
        '''
        
        B, _, N, K = xyz.size()
        scores = xyz  #second torch.Size([1, 6, 15402, 10])

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))
            else:
                scores = self.mlp_convs_nohidden(scores)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden)-1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))
                    else:
                        scores = conv(scores)
                else:
                    if self.bnorm:
                        bn = self.mlp_bns_hidden[i]  #16
                        scores = F.relu(bn(conv(scores)))
                    else:
                        scores = F.relu(conv(scores))

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1)+bias  # B*m*N*K, where bias may bring larger gradient
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores)+bias  # B*m*N*K
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores