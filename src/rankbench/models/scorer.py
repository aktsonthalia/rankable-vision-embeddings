from torch import nn
from torch.nn.functional import cosine_similarity, pairwise_distance
from tqdm import tqdm

from rankbench.utils import *

class Scorer(nn.Module):

    def __init__(self, scorer='cosine_similarity', encoder=None):
        super(Scorer, self).__init__()
        self.scorer = scorer
        self.encoder = encoder
        if scorer == 'cosine_similarity':
            self.scorer = cosine_similarity
        else:
            raise ValueError(f'Unknown scorer: {scorer}')
    
    def forward(self, attribute, dl=None, image_embeddings_cache_path=None, embeddings=None):

        scores = []
        ensemble_prompt = len(attribute) > 1
        attribute = self.encoder.tokenize(attribute)
        attribute = self.encoder.encode_text(attribute)
        if ensemble_prompt:
            attribute = attribute.mean(dim=0, keepdim=True)
        assert attribute.shape[0] == 1
        assert attribute.ndim == 2

        if embeddings is None:
            raise ValueError('embeddings must be provided')
        embed = embeddings.cuda()
        assert embed.ndim == 2
        assert embed.shape[1] == attribute.shape[1]
        scores = self.scorer(embed, attribute).cpu().tolist()
        assert len(scores) == embed.shape[0]
        
        return {
            'scores': scores,
        }