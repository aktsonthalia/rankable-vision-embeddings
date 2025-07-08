import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, 'results')

DATASET_SPLIT_SEED = 42
VAL_SIZE = 0.1

DATASET_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DATASET_PATHS_ARCHIVED = {
    'koniq_10k': {
        'images_dir': os.path.join(DATASET_DIR, 'koniq_10k', 'koniq10k_1024x768'),
    },
    'scenicornot': {
        'images_dir': os.path.join(DATASET_DIR, 'scenicornot', 'images'),
    },
    'celeba': {
        'images_dir': os.path.join(DATASET_DIR, 'celeba'),
        'labels_file': os.path.join(DATASET_DIR, 'compbench_labels/celebA_label/celebA_0_141_rebalanced.json'),
    },
    'fashionpedia': {
        'images_dir': os.path.join(DATASET_DIR, 'fashionpedia'),
        'labels_file': os.path.join(DATASET_DIR, 'compbench_labels/fashion_label/fashion_0_179.json'),
    },
    'mit_states': {
        'images_dir': os.path.join(DATASET_DIR, 'mit-states'),
        'labels_file': os.path.join(DATASET_DIR, 'compbench_labels/st_label/State_Transform_merged0_0_1000.json'),
    },
    'vaw': {
        'images_dir': os.path.join(DATASET_DIR, 'vaw'),
        'labels_file': os.path.join(DATASET_DIR, 'compbench_labels/vaw_label/VAW_0_3133.json'),
    },
    'cub': {
        'images_dir': os.path.join(DATASET_DIR, 'cub'),
        'labels_file': os.path.join(DATASET_DIR, 'compbench_labels/cub_label/CUB_0_1000_rebalanced_fixed_path.json'),
    },
    'fer2013': {
        'images_dir': os.path.join(DATASET_DIR, 'fer2013'),
        'labels_file': os.path.join(DATASET_DIR, 'compbench_labels/fer2013_label/fer2013_0_720_rebalanced.json'),
    },
    'nyu_depth': {
        'images_dir': os.path.join(DATASET_DIR, 'nyu_depth/data'),
        'labels_file': os.path.join(DATASET_DIR, 'compbench_labels/depth_label/depth_0_529_rebalanced.json'),
    },
    'adience': {
        'data_dir': os.path.join(DATASET_DIR, 'adience'),
    },
    'ucf_cc_50': {
        'images_dir': os.path.join(DATASET_DIR, 'ucf_cc_50', 'UCF_CC_50'),
        'num_train': 20,
        'num_val': 5,
        'num_test': 25,
    },
    'lfw10': {
        'images_dir': os.path.join(DATASET_DIR, 'lfw10', 'LFW10', 'images'),
        'annotations_dir': os.path.join(DATASET_DIR, 'lfw10', 'LFW10', 'annotations'),
    },
    'foggy_cityscapes': {
        'images_dir': os.path.join(DATASET_DIR, 'foggy_cityscapes', 'leftImg8bit_foggyDBF'),
    },
    'ut_zap50k': {
        'images_dir': os.path.join(DATASET_DIR, 'utzap50k-1', 'ut-zap50k-images-square'),
        'labels_file': os.path.join(DATASET_DIR, 'utzap50k-1', 'ut-zap50k-data', 'zappos-labels.mat'),
        'image_paths_file': os.path.join(DATASET_DIR, 'utzap50k-1', 'ut-zap50k-data', 'image-path.mat'),
    },
    'bone_age': {
        'images_dir': os.path.join(DATASET_DIR, 'bone_age'),
        'labels_dir': os.path.join(DATASET_DIR, 'bone_age', 'labels'),
    },
    'digits': {
        'images_dir': os.path.join(DATASET_DIR, 'digits'),
    },
    '2dimage2bmi': {
        'images_dir': os.path.join(DATASET_DIR, '2dimage2bmi', 'datasets'),
    },
    'ripeness': {
        'images_dir': os.path.join(DATASET_DIR, 'ripeness', 'Dataset'),
    },
}

DATASET_PATHS = {
    'nwpu_crowd': {
        'data_dir': os.path.join(DATASET_DIR, 'nwpu_crowd'),
    },
    'shanghaitech': {
        'data_dir': os.path.join(DATASET_DIR, 'shanghaitech'),
    },
    'lfw': {
        'data_dir': os.path.join(DATASET_DIR, 'lfw10', 'LFW10'),
    },
    'adience': {
        'data_dir': os.path.join(DATASET_DIR, 'adience'),
    },
    'ava': {
        'data_dir': os.path.join(DATASET_DIR, 'ava'),
    },
    'ucf_qnrf': {
        'data_dir': os.path.join(DATASET_DIR, 'ucf_qnrf', 'UCF-QNRF_ECCV18'),
    },
    'utkface': {
        'data_dir': os.path.join(DATASET_DIR, 'utkface'),
        'train_split_file': os.path.join(DATASET_DIR, 'utkface', 'splits', 'utk_train.csv'),
        'test_split_file': os.path.join(DATASET_DIR, 'utkface', 'splits', 'utk_test.csv'),
    },
    'hci': {
        'data_dir': os.path.join(DATASET_DIR, 'hci', 'HistoricalColor-ECCV2012', 'data', 'imgs', 'decade_database'),
    },
    'kinect': {
        'data_dir': os.path.join(DATASET_DIR, 'kinect'),
    },
    'awa2': {
        'data_dir': os.path.join(DATASET_DIR, 'awa2'),
    },
    'koniq_10k': {
        'data_dir': os.path.join(DATASET_DIR, 'koniq_10k'),
    },
}
for dataset in DATASET_PATHS:
    DATASET_PATHS[dataset]['embeddings_dir'] = os.path.join(DATASET_PATHS[dataset]['data_dir'], 'embeddings')

SOTA_NUMBERS = {
    'utkface': {
        'paper': 'https://openreview.net/forum?id=KbCh7zbw2K',
        'metrics': {
            'mae': 3.83
        }
    },
    'hci': {
        'paper': 'https://arxiv.org/pdf/2307.04616v2',
        'metrics': {
            'mae': 0.32
        }
    },
    'ava': {
        'paper': 'https://arxiv.org/pdf/2307.04616v2',
        'metrics': {
            'spearman': 0.747,
        }
    },
    'adience': {
        'paper': 'https://arxiv.org/pdf/2307.04616v2',
        'metrics': {
            'mae': 0.36,
        }
    },
    'ucf_qnrf': {
        'paper': 'https://arxiv.org/pdf/2403.09281v3',
        'metrics': {
            'mae': 75.9,
        }
    }
}

SOTA_NUMBERS_WITH_MORE_DATA = {
    'utkface': {
        'paper': 'https://arxiv.org/pdf/2307.04616v2',
        'metrics': {
            'mae': 3.700
        }
    },
}

PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')

PROMPTS = {
    'ava': {
        'good_looking_photo': [
            'a picture with a high aesthetic appeal',
        ],
    },
    'utkface': {
        'age': [
            'a a professional photo of a 80 year old 60 years old middle aged 70 years old',
            # 'a picture of an aged person',
            # 'a portrait of an aged person',
        ],
    },
    'adience': {
        'age': [
            'a picture of an aged person',
        ],
    },
    'hci': {
        'clicked_recently': [
            'a recently clicked photo',
        ],
    },
    'lfw10': {
        'smile': [
            'a picture of a smiling face',
            'a picture of a person smiling',
        ],
        'young': [
            'a picture of a young person',
            'a picture of a person who looks young',
        ],
        'v_teeth': [
            'a picture of a person showing their teeth',
            'a picture of a person with visible teeth',
        ],
        'masculinelooking': [
            'a picture of a person with a masculine look',
            'a picture of a person who looks masculine',
        ],
        'darkhair': [
            'a picture of a person with dark hair',
            'a picture of a person with black hair',
        ],
        'baldhead': [
            'a picture of a person with a bald head',
            'a picture of a person who is bald',
        ],
        'eyesopen': [
            'a picture of a person with their eyes open',
            'a picture of a person who has their eyes open',
        ],
        'mouthopen': [
            'a picture of a person with their mouth open',
            'a picture of a person who has their mouth open',
        ],
        'vforehead': [
            'a picture of a person with their forehead visible',
            'a picture of a person with a visible forehead',
        ],
        'goodlooking': [
            'a picture of a good looking person',
            'a picture of a person who looks good',
        ],
    },
    'ucf_cc_50': {  
        'crowd_count': [
            'a picture with many people',
            'a picture with a large crowd',
        ],
    },
    'ucf_qnrf': {  
        'crowd_count': [
            'a picture with many people',
            'a picture with a large crowd',
        ],
    },
    'foggy_cityscapes': {
        'fog': [
            'a picture with fog',
            'a picture with foggy weather',
        ],
    },
    'ut_zap50k': {
        'open': [
            'a picture of an open-toe shoe',
            'a picture of a shoe with an open toe',
        ],
        'pointy': [
            'a picture of a pointy shoe',
            'a picture of a shoe with a pointy toe',
        ],
        'sporty': [
            'a picture of a sporty shoe',
            'a picture of a shoe for sports',
        ],
        'comfort': [
            'a picture of a comfortable shoe',
            'a picture of a shoe that is comfortable',
        ],
    },
    'bone_age': {
        'boneage': [
            'a picture of an x-ray of an aged hand bone',
        ],
    },
    'digits': {
        'high_number': [
            'a picture of a hand indicating a high number',
        ]
    },
    '2dimage2bmi': {    
        'age': [
            'a picture of an aged person',
        ],
        'height': [
            'a picture of a tall person',
        ],
        'weight': [
            'a picture of a person with a high body weight',
        ],
    },
    'ripeness': {
        'decay': [
            'a picture of a mature fruit',
            'a picture of a fruit in an advanced stage of ripeness',
        ],
    },
}

META_RESULTS_FILE = 'meta_results.json'