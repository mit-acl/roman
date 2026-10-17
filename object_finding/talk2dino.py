"""Text embeddings in the space of ROMAN's DINOv2 descriptors, via Talk2DINO.

Talk2DINO (arXiv 2411.19331) learns a small MLP that maps CLIP ViT-B/16 text embeddings
into the patch-token space of DINOv2 ViT-B/14 with registers. ROMAN's segment descriptors
are averages of those patch tokens inside each mask, so a projected text embedding can be
compared to them directly by cosine similarity, for a map built with
dinov2-with-registers-base. Only the text side is loaded here: the CLIP text encoder and
projection weights come from the lorebianchi98/Talk2DINO-ViTB checkpoint, following its
build_text_embedding (average the raw CLIP embeddings over prompt templates, project,
normalize).
"""
import clip
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# sub_imagenet_template from Talk2DINO/FreeDA templates.py
TEMPLATES = ['itap of a {}.', 'a bad photo of a {}.', 'a origami {}.', 'a photo of the large {}.',
             'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']


class Talk2DINOText:
    def __init__(self, device):
        path = hf_hub_download('lorebianchi98/Talk2DINO-ViTB', 'model.safetensors')
        with safe_open(path, 'pt') as f:
            clip_sd = {k[len('clip_model.'):]: f.get_tensor(k) for k in f.keys() if k.startswith('clip_model.')}
            proj = {k[len('proj.'):]: f.get_tensor(k).float().to(device) for k in f.keys() if k.startswith('proj.')}
        self.clip = clip.model.build_model(clip_sd).float().eval().to(device)
        self.proj = proj
        self.device = device

    @torch.no_grad()
    def encode(self, texts, templates=TEMPLATES):
        out = []
        for text in texts:
            tokens = clip.tokenize([t.format(text) for t in templates]).to(self.device)
            x = self.clip.encode_text(tokens).float().mean(dim=0)  # raw CLIP embeddings averaged over templates
            x = x @ self.proj['linear_layer.weight'].T + self.proj['linear_layer.bias']
            x = torch.tanh(x) @ self.proj['hidden_layers.0.weight'].T + self.proj['hidden_layers.0.bias']
            out.append(torch.nn.functional.normalize(x, dim=0).cpu().numpy())
        return np.stack(out)
