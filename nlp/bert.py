import torch.nn as nn

class BERT(nn.Module):
    """
    A multi-layer bidirectional Transformer encoder.

    Summary:
    •   Pre-trained language representations: Feature-based (e.g., ELMo) vs Fine-tuning (e.g., GPT)
    •   BERT = "Bidirectional" (from ELMo) + "Transformer" (~GPT, but BERT uses encoders)
    •   Crucial to incorporate context from both directions (bi-directional)
    •   Tasks:
        1. MLM (masked language model): Fill in the blanks (masked tokens)
        2. NSP (next sentence prediction): Determine if two sentences are adjacent -> 50% positive / 50% negative
    •   Pre-training (unlabeled) -> Fine-tuning (labeled)
    •   Total Parameters = 30k * H + L * H^2 * 12
    •   WordPiece embeddings with first token as [CLS] (classification) and last token as [SEP] (separator)
    •   Input = Token embeddings + Segment embeddings (sentence A or B) + Position embeddings (0, 1, 2, ...)
    •   Mismatch between pre-training and fine-tuning ([MASK] not in fine-tuning) -> 80% [MASK] / 10% random / 10% unchanged
    •   Should fine-tune with more epochs and a better optimizer
    •   Encoder: Not good at generative tasks (e.g., machine translation), but good at discriminative tasks (e.g., classification)

    Interesting Stories:
    •   Bert (BERT) is a character from Sesame Street, same as Elmo (ELMo)

    References:
    https://arxiv.org/pdf/1810.04805.pdf
    https://github.com/google-research/bert
    """

    # TODO: Create BERT and related classes
