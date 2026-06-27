import timm
import torch.nn as nn


def build_vit(
    num_classes: int,
    pretrained: bool = True,
    model_name: str = "vit_base_patch16_224",
    drop_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Builds a Vision Transformer model using the timm library.

    Args:
        num_classes:     Number of output classes (automatically inferred from the dataset).
        pretrained:      If True, initializes weights from ImageNet pretraining.
        model_name:      Name of the timm model to use.
        drop_rate:       Dropout rate applied after the attention and MLP layers.
        drop_path_rate:  Stochastic depth rate for regularization during training.
        freeze_backbone: If True, only the classification head is trainable.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    return model