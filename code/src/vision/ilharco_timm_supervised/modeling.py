import os
from typing import Optional

import timm
import timm.data
import torch


class ImageEncoder(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()

        print(f"Loading {model_name} pre-trained weights.")
        self.model_name = model_name

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.train_preprocess = timm.data.create_transform(**data_config, is_training=True)
        self.val_preprocess = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, images):
        return self.model(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)

    # @classmethod
    # def load(
    #     cls,
    #     model_name: str,
    #     filename: str,
    #     map_location: str = "cpu",
    # ):
    #     print(f"Loading image encoder from {filename}")
    #     obj = cls(model_name=model_name)
    #     state_dict = torch.load(filename, map_location=map_location)
    #     obj.load_state_dict(state_dict)
    #     return obj


# class ClassificationHead(torch.nn.Linear):
#     def __init__(self, normalize: bool, weights: torch.Tensor, biases: Optional[torch.Tensor] = None):
#         output_size, input_size = weights.shape
#         super().__init__(input_size, output_size)
#         self.normalize = normalize
#         # Stored as a buffer so it round-trips through state_dict.
#         self.register_buffer("normalize_flag", torch.tensor(bool(normalize)))
#         if weights is not None:
#             self.weight = torch.nn.Parameter(weights.clone())
#         if biases is not None:
#             self.bias = torch.nn.Parameter(biases.clone())
#         else:
#             self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

#     def forward(self, inputs):
#         if bool(self.normalize_flag):
#             inputs = inputs / inputs.norm(dim=-1, keepdim=True)
#         return super().forward(inputs)

#     def __call__(self, inputs):
#         return self.forward(inputs)

#     def save(self, filename):
#         print(f"Saving classification head to {filename}")
#         if os.path.dirname(filename):
#             os.makedirs(os.path.dirname(filename), exist_ok=True)
#         torch.save(self.state_dict(), filename)

#     @classmethod
#     def load(cls, filename: str, map_location: str = "cpu"):
#         print(f"Loading classification head from {filename}")
#         state_dict = torch.load(filename, map_location=map_location)
#         weights = state_dict["weight"]
#         biases = state_dict["bias"]
#         normalize = bool(state_dict.get("normalize_flag", torch.tensor(False)))
#         obj = cls(normalize=normalize, weights=weights, biases=biases)
#         obj.load_state_dict(state_dict)
#         return obj


# class ImageClassifier(torch.nn.Module):
#     def __init__(self, image_encoder, classification_head):
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.classification_head = classification_head
#         if self.image_encoder is not None:
#             self.train_preprocess = self.image_encoder.train_preprocess
#             self.val_preprocess = self.image_encoder.val_preprocess

#     def freeze_head(self):
#         self.classification_head.weight.requires_grad_(False)
#         self.classification_head.bias.requires_grad_(False)

#     def forward(self, inputs):
#         features = self.image_encoder(inputs)
#         outputs = self.classification_head(features)
#         return outputs

#     def __call__(self, inputs):
#         return self.forward(inputs)

#     def save(self, filename):
#         print(f"Saving image classifier to {filename}")
#         if os.path.dirname(filename):
#             os.makedirs(os.path.dirname(filename), exist_ok=True)
#         torch.save(self.state_dict(), filename)

#     @classmethod
#     def load(
#         cls,
#         model_name: str,
#         filename: str,
#         map_location: str = "cpu",
#     ):
#         print(f"Loading image classifier from {filename}")
#         state_dict = torch.load(filename, map_location=map_location)

#         head_weight = state_dict["classification_head.weight"]
#         head_bias = state_dict["classification_head.bias"]
#         head_normalize = bool(state_dict.get("classification_head.normalize_flag", torch.tensor(False)))

#         image_encoder = ImageEncoder(model_name=model_name)
#         classification_head = ClassificationHead(
#             normalize=head_normalize, weights=head_weight, biases=head_bias
#         )
#         obj = cls(image_encoder, classification_head)
#         obj.load_state_dict(state_dict)
#         return obj




if __name__ == "__main__":

    from rich.pretty import pprint

    model = ImageEncoder(
        model_name="timm/vit_base_patch16_clip_224.openai_ft_in12k_in1k",
        num_classes=23
    )
    
    pprint(model, expand_all=True)