import torch


def index_select(x, idx):
    """
    input :     x -> [N, F, 1]     F is feature dimension
                batch_idx -> [N]
                idx -> [M, K]
    output :    selected -> [M, F, K]
    """
    N, F = x.shape[:2]
    M, K = idx.shape
    idx = idx.contiguous().view(-1)
    selected = x.transpose(2, 1)[idx]
    selected = selected.view(M, K, F).permute(0, 2, 1).contiguous()
    return selected

def pointwise_distance(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    #x torch.Size([28087, 3, 1])
    #y torch.Size([28087, 3, 1])
    with torch.no_grad():
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        x = x.unsqueeze(-1)                 #torch.Size([23186, 3, 1])
        y = y.transpose(0,1).unsqueeze(0)   #torch.Size([1, 3, 23186])
        diff = x - y                        #torch.Size([23186, 3, 23186])
        dis = torch.sum(torch.square(diff), dim=1)
        if torch.min(dis) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dis    #[23186 23186]
        else:
            return torch.sqrt(dis)

def get_dists(points1,square=True):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(M, C)
    :param points: shape=(N, C)
    '''
    with torch.no_grad():
        # points1.squeeze(-1)
        # points2.squeeze(-1)
        # B = 1
        M, C = points1.shape
        # N, _ = points2.shape
        a = torch.sum(torch.pow(points1, 2), dim=-1).view(M, 1)
        dists = a+a.T
        # dists = torch.sum(torch.pow(points1, 2), dim=-1).view(M, 1) + \
        #         torch.sum(torch.pow(points1, 2), dim=-1).view(1, M)
        # dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
        # print(1)
        # points2.permute(1,0)
        # dists -= 2 * torch.matmul(points1, points2.permute(1,0))
        dists -= 2 * torch.mm(points1, points1.T)

        # dists.squeeze(0)
        dists = dists.squeeze(0).int()

        # dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
        if torch.min(dists) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dists
        else:
            return torch.sqrt(dists)


def pointwise_distance_copy2(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    #x torch.Size([28087, 3, 1])
    #y torch.Size([28087, 3, 1])
    with torch.no_grad():
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        x = x.unsqueeze(-1)                 #torch.Size([23186, 3, 1])
        y = y.transpose(0,1).unsqueeze(0)   #torch.Size([1, 3, 23186])
        print(x.shape)
        diff = x - y                        #torch.Size([23186, 3, 23186])
        d1 = diff.chunk(20000,0)
        list2 = []
        for _ in d1:
            # list2.append(torch.sum(torch.square(_), dim=1))
            list2.append(torch.topk(torch.sum(torch.square(_), dim=1),k=9,largest=False)[1])
       
        dis = torch.cat(list2,dim=0)
        #清除显存
        # torch.cuda.empty_cache()

        # dis = torch.sum(torch.square(diff), dim=1)
        if torch.min(dis) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dis
        else:
            return torch.sqrt(dis)

def pointwise_distance_copy4(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    #x torch.Size([28087, 3, 1])
    #y torch.Size([28087, 3, 1])
    with torch.no_grad():
        # x_ = x.chunk(10000,0)
        # y_ = y.chunk(10000,0)

        # for xi in x_:
        x = x.squeeze(-1)
            

        y = y.squeeze(-1)


        # x = x.unsqueeze(-1)                 #torch.Size([23186, 3, 1])
        y = y.transpose(0,1).unsqueeze(0)   #torch.Size([1, 3, 23186])
        print(x.shape)
        
        x_ = x.chunk(10000,0)
        y_ = y.chunk(10000,2)
        diff_ = []
        for i in x_:
            diff_.append = i - y
        diff = torch.cat(diff_,dim=0)

        # diff = x - y                        #torch.Size([23186, 3, 23186])



        d1 = diff.chunk(10000,0)
        list2 = []
        for _ in d1:
            # list2.append(torch.sum(torch.square(_), dim=1))
            list2.append(torch.topk(torch.sum(torch.square(_), dim=1),k=9,largest=False)[1])
       
        dis = torch.cat(list2,dim=0)
        #清除显存
        # torch.cuda.empty_cache()

        # dis = torch.sum(torch.square(diff), dim=1)
        if torch.min(dis) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dis
        else:
            return torch.sqrt(dis)


def pointwise_distance_copy3(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    #x torch.Size([28087, 3, 1])
    #y torch.Size([28087, 3, 1])
    # k = 16
    import numpy as np 
    with torch.no_grad():
        res = []

        for i in x: 
            dists = []
            for j in y :
                dists.append(torch.sum((i-j)**2))

            # idxs = torch.cat(torch.Tensor(dists),dim=0)
            idxs = torch.Tensor(dists)
            
            res.append(idxs)


        x = x.squeeze(-1)
        y = y.squeeze(-1)

        x = x.unsqueeze(-1)                 #torch.Size([23186, 3, 1])
        y = y.transpose(0,1).unsqueeze(0)   #torch.Size([1, 3, 23186])
        diff = x - y                        #torch.Size([23186, 3, 23186])
        dis = torch.sum(torch.square(diff), dim=1)
        if torch.min(dis) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dis
        else:
            return torch.sqrt(dis)


def pointwise_distance_copy(x, y, square=True):
    """
    The pointwise distance from x to y
    """
    #x torch.Size([28087, 3, 1])
    #y torch.Size([28087, 3, 1])
    with torch.no_grad():
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        x = x.unsqueeze(-1)                 #torch.Size([23186, 3, 1])
        y = y.transpose(0,1).unsqueeze(0)   #torch.Size([1, 3, 23186])
        diff = x - y                        #torch.Size([23186, 3, 23186])

        # dis = torch.sum(torch.square(diff), dim=1)
    
        dis0 = []
        for i in diff:
            dis0.append(torch.square(i).sum(1))

        dis = torch.cat(dis0,dim=0)

        if torch.min(dis) < 0:
            raise RuntimeError('dis small than 0')
        if square:
            return dis
        else:
            return torch.sqrt(dis)


def knn1(x, batch_idx, k:int=16):
    """
    input :     x -> [N, F, 1]      #torch.Size([45964, 3, 1])
                batch_idx -> [N]    # torch.Size([45964])
                k -> int            # k =9 
    output :    index -> [N, K]
    """
    with torch.no_grad():
        batch_size = torch.max(batch_idx) + 1      #8
        index_base = torch.zeros([x.shape[0], 1], dtype=torch.long, device=x.device) #torch.Size([45964, 1])
        index_list = []
        base = 0
        for bs in range(batch_size):
            x_bs = x[batch_idx==bs]  #6946, 3, 1
            dis = get_dists(x_bs.squeeze(-1).detach().float())   #6946, 6946  
            # dis = pointwise_distance(x_bs.detach(), x_bs.detach())   #6946, 6946  

            _, idx = torch.topk(-dis, k=k)  # torch.Size([6946, 9])  筛选每一列中距离最近的k个
            index_list.append(idx) 
            index_base[batch_idx==bs] = base  #0
            base += len(x_bs)   #6946
            # print(index_base)
        index = torch.cat(index_list, dim=0) + index_base  #torch.Size([45964, 9])
    return index


def knn_copy(x, batch_idx, k:int=16):
    """
    input :     x -> [N, F, 1]      #torch.Size([45964, 3, 1])
                batch_idx -> [N]    # torch.Size([45964])
                k -> int            # k =9 
    output :    index -> [N, K]
    """
    with torch.no_grad():
        batch_size = torch.max(batch_idx) + 1      #8
        index_base = torch.zeros([x.shape[0], 1], dtype=torch.long, device=x.device) #torch.Size([45964, 1])
        index_list = []
        base = 0
        for bs in range(batch_size):
            x_bs = x[batch_idx==bs]  #6946, 3, 1
            # dis = pointwise_distance_copy2(x_bs.detach(), x_bs.detach()) 
            dis = get_dists(x_bs.detach(), x_bs.detach()) 
                
            # dis = pointwise_distance(x_bs.detach(), x_bs.detach())   #6946, 6946   # x_bs  6946,3, 1

            # _, idx = torch.topk(-dis, k=k)  # torch.Size([6946, 9])  筛选每一列中距离最近的k个

            # index_list.append(idx) 
            index_list.append(dis) 

            index_base[batch_idx==bs] = base  #0
            base += len(x_bs)   #6946
            # print(index_base)
        index = torch.cat(index_list, dim=0) + index_base  #torch.Size([45964, 9])
    return index



def knn(x, batch_idx, k:int=16,frame_id=0):
# def knn(x, batch_idx, k:int=16):

    """
    input :     x -> [N, F, 1]      #torch.Size([45964, 3, 1])
                batch_idx -> [N]    # torch.Size([45964])
                k -> int            # k =9 
    output :    index -> [N, K]
    """
    with torch.no_grad():
        batch_size = torch.max(batch_idx) + 1      #8
        index_base = torch.zeros([x.shape[0], 1], dtype=torch.long, device=x.device) #torch.Size([45964, 1])
        index_list = []
        base = 0
        for bs in range(batch_size):
            x_bs = x[batch_idx==bs]  #6946, 3, 1
            dis = get_dists(x_bs.squeeze(-1).detach().float())   #6946, 6946  

            # #保存距离矩阵 画图
            # dis = pointwise_distance(x_bs.detach(), x_bs.detach(),False)   #6946, 6946  
            # torch.save(dis,'/home/jiangchen/work/charm/svae_pt/dis%s.pt'%frame_id)
            # print('distance torch%s.save'%frame_id)

            _, idx = torch.topk(-dis, k=k)  # torch.Size([6946, 9])  筛选每一列中距离最近的k个
            index_list.append(idx) 
            index_base[batch_idx==bs] = base  #0
            base += len(x_bs)   #6946
        index = torch.cat(index_list, dim=0) + index_base  #torch.Size([45964, 9])
    return index

