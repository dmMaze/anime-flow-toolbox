import sys
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


import math
import torch
import torch.nn.functional as F

from utils.torch_utils import img2tensor, tensor2img, init_model_from_pretrained
from ..flow_utils import flow_to_image, flow2rgb, InputPadder, resize_flow
from .cfg import get_cfg, build_flowformer

TRAIN_SIZE = [432, 960]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights


def compute_flow(model, image1, image2, weights=None, output_type='numpy'):

    image_size = image1.shape[-2:]

    # image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count

    if output_type == 'numpy':
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:
        flow = flow_pre

    return flow


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size


def build_flow_model():
    cfg = get_cfg()
    cfg.latentcostformer.pretrain = False
    model = build_flowformer(cfg)
    return model

flowformer = None

@torch.inference_mode()
def apply_flow_former(image1, image2, keep_size=False, device='cuda'):

    if not isinstance(image1, torch.Tensor):
        image1 = img2tensor(image1, device=device)
    if not isinstance(image2, torch.Tensor):
        image2 = img2tensor(image2, device=device)

    h, w = image1.shape[-2:]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = torch.nn.functional.interpolate(image1, size=(dsize[1], dsize[0]), mode='bicubic')
        image2 = torch.nn.functional.interpolate(image2, size=(dsize[1], dsize[0]), mode='bicubic')

    global flowformer
    if flowformer is None:
        flowformer = init_model_from_pretrained(
            'dreMaz/AnimeRun_ckpt', build_flow_model, weights_name='flowformer_sintel.safetensors', device=device
        ).eval()

    flow = compute_flow(flowformer, image1, image2, output_type='tensor')
    fh, fw = flow.shape[-2:]
    flow = flow[0].permute(1, 2, 0).cpu().numpy()

    if fh != h or fw != w:
        flow, _ = resize_flow(flow, (w, h))

    return flow