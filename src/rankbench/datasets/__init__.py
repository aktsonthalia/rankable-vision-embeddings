from functools import partial

from .base import PairwiseComparisonDataset
from .utkface import UTKFace
from .ucf_qnrf import UCF_QNRF
from .hci import HCI
from .ava import AVA
from .adience import Adience
from .kinect import Kinect
from .koniq_10k import Koniq10k
from .shanghaitech import ShanghaiTech

datasets = {
    'adience': Adience,
    'ucf_qnrf': UCF_QNRF,
    'utkface': UTKFace,
    'hci': HCI,
    'ava': AVA,
    'kinect': Kinect,
    'koniq_10k': Koniq10k,
    'shanghaitech_a': partial(ShanghaiTech, part='A'),
    'shanghaitech_b': partial(ShanghaiTech, part='B'),
}
