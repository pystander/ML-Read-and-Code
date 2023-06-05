import torch
import torch.nn as nn
from ..nlp.transformer import Encoder

class ViT(nn.Module):
    """
    A Transformer-based model for image recognition. Performs CV tasks in NLP-like approach.

    Summary:
    •   Not necessarily reply on CNNs for image classification
    •   Pre-train on large corpora and fine-tune on smaller datasets (~ BERT, GPT)
    •   Sequence of image pixels would be too long (e.g., N = 224 x 224 = 50176)
        Solution: Split image into patches (e.g., 16 x 16 patches, N = (224 / 16) * (224 / 16) = 196)
    •   Train model in supervised manner
    •   Transformers lack inductive biases (e.g., locality, translation equivariance)
        Large scale training "trumps" inductive bias
    •   ViT = Flattened Patches -> Patch + Position Embeddings -> Transformer Encoder -> MLP Head -> Class

    Interesting Stories:
    •   Few weeks later, ViT was applied in detection (ViT-FRCNN) and segmentation (SETR)

    References:
    https://arxiv.org/pdf/2010.11929.pdf
    https://github.com/google-research/vision_transformer
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    """

    # TODO: Create ViT and related classes
