'''
Descripttion: your project
version: 1.0
Author: Chen Jiang
Date: 2021-08-20 21:58:32
LastEditTime: 2021-08-20 21:58:33
'''
from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
# from .point_rcnn import PointRCNN_gcn
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .pv_rcnn_ssl import PVRCNN_SSL
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .pv_rcnn_gcn import PVRCNN_gcn

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PVRCNN_SSL': PVRCNN_SSL,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PVRCNN_gcn': PVRCNN_gcn,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    # 'PointRCNN_gcn': PointRCNN_gcn,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN':VoxelRCNN,
    'SATgcn':PointPillar,


}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
