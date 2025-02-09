import torch
import torch.nn as nn
import open_clip

class VisionLanguageCaptioningModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B-32", pretrained_clip=True, vocab_size=50257, hidden_dim=512):
        super().__init__()

        # Load CLIP as encoder
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained="openai" if pretrained_clip else None
        )
        self.clip_model.visual.train()  # Fine-tune CLIP visual layers

        # Decoder: Transformer for language generation
        self.decoder = nn.Transformer(
            d_model=hidden_dim,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
        )

        # Output layer to generate tokens
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image, input_ids, attention_mask):
        # Extract image features from CLIP
        image_features = self.clip_model.encode_image(image).unsqueeze(1)

        # Embed tokens and pass through Transformer decoder
        output = self.decoder(image_features, input_ids.unsqueeze(1))
        logits = self.output_layer(output)
        return logits
