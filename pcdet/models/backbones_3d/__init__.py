from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG,PointNet2MSG_gcn
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x,VoxelBackBone8x_gcn
from .spconv_unet import UNetV2
from .pointnet2_backbone_gcn import PointNet2MSG_gcn2
from .gcn.PlainGCN import PlainGCN

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'PointNet2MSG_gcn': PointNet2MSG_gcn,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelBackBone8x_gcn': VoxelBackBone8x_gcn,

    'PointNet2MSG_gcn2': PointNet2MSG_gcn2,

    'SATgcn': PlainGCN

}
