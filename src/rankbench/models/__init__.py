from functools import partial

from .clip_old import OpenAICLIP
from .comparator import *
from .scorer import Scorer


from rankbench.models.open_clip import OpenCLIP
from rankbench.models.timm_models import TimmModel

models = {
    "open_clip_convnext_large_d_320": partial(OpenCLIP, variant="convnext_large_d_320"),
    "openai_clip_vit_b_32": partial(OpenAICLIP, variant="ViT-B/32"),
    "openai_clip_vit_b_16": partial(OpenAICLIP, variant="ViT-B/16"),
    "openai_clip_vit_l_14": partial(OpenAICLIP, variant="ViT-L/14"),
    "openai_clip_vit_l_14_336": partial(OpenAICLIP, variant="ViT-L/14@336px"),
    "openai_clip_vit_h_14": partial(OpenAICLIP, variant="ViT-H/14"),
    "openai_clip_rn50": partial(OpenAICLIP, variant="RN50"),
    "openai_clip_rn50x4": partial(OpenAICLIP, variant="RN50x4"),
    "openai_clip_rn101": partial(OpenAICLIP, variant="RN101"),
    "cosine_comparator": CosineSimilarityComparator,
    "trainable_mlp_comparator": TrainableMLPComparator,
    "linear_projection_comparator": LinearProjectionComparator,
    "non_linear_projection_comparator": NonLinearProjectionComparator,
    "scorer_cosine_similarity": partial(Scorer, scorer='cosine_similarity'),
    "timm_resnet50": partial(TimmModel, variant="resnet50"),
    "timm_vit_base_patch14_dinov2": partial(TimmModel, variant="vit_base_patch14_dinov2"),
    "timm_vit_base_patch32_224": partial(TimmModel, variant="vit_base_patch32_224"),
    "timm_convnextv2_large": partial(TimmModel, variant="convnextv2_large"),
}