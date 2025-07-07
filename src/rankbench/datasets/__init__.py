from functools import partial

from .base import PairwiseComparisonDataset
from .utkface import UTKFace
from .ucf_cc_50 import UCFCC50Dataset
from .lfw import LFW10
from .foggy_cityscapes import FoggyCityscapes
from .ut_zappos50k import UTZap50kDataset
from .bone_age import BoneAge
from .digits import DigitsDataset
from .image2bmi import Image2BMIDataset
from .ripeness import PalmFruitRipeness
from .ucf_qnrf import UCF_QNRF
from .hci import HCI
from .ava import AVA
from .adience import Adience
from .kinect import Kinect
from .awa2 import AwA2
from .koniq_10k import Koniq10k
from .shanghaitech import ShanghaiTech
from .nwpu_crowd import NWPU_Crowd

datasets = {
    'lfw': LFW10,
    'adience': Adience,
    'ucf_qnrf': UCF_QNRF,
    'utkface': UTKFace,
    'hci': HCI,
    'ava': AVA,
    'kinect': Kinect,
    'awa2': AwA2,
    'koniq_10k': Koniq10k,
    'shanghaitech_a': partial(ShanghaiTech, part='A'),
    'shanghaitech_b': partial(ShanghaiTech, part='B'),
    'nwpu_crowd': NWPU_Crowd,
}
