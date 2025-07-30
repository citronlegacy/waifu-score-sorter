# Code copied directly from the original waifu-scorer repository: https://huggingface.co/spaces/Eugeoter/waifu-scorer-v3/blob/main/utils.py
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Union
from PIL import Image

WS_REPOS = ["Eugeoter/waifu-scorer-v3"]


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating', batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048) if batch_norm else nn.Identity(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512) if batch_norm else nn.Identity(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if batch_norm else nn.Identity(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


class WaifuScorer(object):
    def __init__(self, model_path=None, device='cuda', cache_dir=None, verbose=False):
        self.verbose = verbose
        if model_path is None:
            model_path = repo2path(WS_REPOS[0])
            if self.verbose:
                print(f"model path not set, switch to default: `{model_path}`")
        if not os.path.isfile(model_path):
            model_path = download_from_url(model_path, cache_dir=cache_dir)

        print(f"loading pretrained model from `{model_path}`")
        self.mlp = load_model(model_path, input_size=768, device=device)
        self.model2, self.preprocess = load_clip_models("ViT-L/14", device=device)
        self.device = self.mlp.device
        self.dtype = self.mlp.dtype
        self.mlp.eval()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image]) -> Union[List[float], float]:
        if isinstance(images, Image.Image):
            images = [images]
        n = len(images)
        if n == 1:
            images = images*2  # batch norm
        images = encode_images(images, self.model2, self.preprocess, device=self.device).to(device=self.device, dtype=self.dtype)
        predictions = self.mlp(images)
        scores = predictions.clamp(0, 10).cpu().numpy().reshape(-1).tolist()
        # if n == 1:
        #     scores = scores[0]
        return scores


def repo2path(model_repo_and_path: str):
    if os.path.isfile(model_repo_and_path):
        model_path = model_repo_and_path
    elif os.path.isdir(model_repo_and_path):
        model_path = os.path.join(model_repo_and_path, "model.pth")
    elif model_repo_and_path in WS_REPOS:
        model_path = model_repo_and_path + '/model.pth'
    else:
        raise ValueError(f"Invalid model_repo_and_path: {model_repo_and_path}")
    return model_path


def download_from_url(url, cache_dir=None, verbose=True):
    from huggingface_hub import hf_hub_download
    split = url.split("/")
    username, repo_id, model_name = split[-3], split[-2], split[-1]
    # if verbose:
    # print(f"[download_from_url]: {username}/{repo_id}/{model_name}")
    model_path = hf_hub_download(f"{username}/{repo_id}", model_name, cache_dir=cache_dir)
    return model_path


def load_clip_models(name: str = "ViT-L/14", device='cuda'):
    import clip
    model2, preprocess = clip.load(name, device=device)  # RN50x64
    return model2, preprocess


def load_model(model_path: str = None, input_size=768, device: str = 'cuda', dtype=None):
    model = MLP(input_size=input_size)
    if model_path:
        s = torch.load(model_path, map_location=device)
        model.load_state_dict(s)
        model.to(device)
    if dtype:
        model = model.to(dtype=dtype)
    return model


def normalized(a: torch.Tensor, order=2, dim=-1):
    l2 = a.norm(order, dim, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


@torch.no_grad()
def encode_images(images: List[Image.Image], model2, preprocess, device='cuda') -> torch.Tensor:
    if isinstance(images, Image.Image):
        images = [images]
    image_tensors = [preprocess(img).unsqueeze(0) for img in images]
    image_batch = torch.cat(image_tensors).to(device)
    image_features = model2.encode_image(image_batch)
    im_emb_arr = normalized(image_features).cpu().float()
    return im_emb_arr
