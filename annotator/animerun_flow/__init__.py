import torch
from argparse import Namespace

from .core.gmflow.gmflow import GMFlow
from .core.raft import RAFT
from ..flow_utils import InputPadder
from utils.torch_utils import img2tensor, tensor2img, init_model_from_pretrained
import numpy as np


URL_GMFLOW = 'https://huggingface.co/dreMaz/AnimeRun_ckpt/resolve/main/20000_gmflow-animerun-v2-ft.pth'
URL_RAFT = 'https://huggingface.co/dreMaz/AnimeRun_ckpt/resolve/main/20000_raft-animerun-v2-ft_again.pth'
 
model_gmflow: GMFlow = None
@torch.inference_mode()
def apply_gmflow(image1, image2, device='cuda', unload=False, inference_size=384, output_type='numpy'):

    '''
    training size should be 384 https://github.com/haofeixu/gmflow/blob/b5123431164d01ec14526a1c3d22218aecb62024/main.py#L28
    '''

    global model_gmflow

    if model_gmflow is None:
        model_gmflow_args = dict(feature_channels=128,
                    num_scales=2,
                    upsample_factor=4,
                    num_head=1,
                    attention_type='swin',
                    ffn_dim_expansion=4,
                    num_transformer_layers=6,
                    )
        model_gmflow = init_model_from_pretrained(
            'dreMaz/AnimeRun_ckpt', GMFlow, model_args=model_gmflow_args, 
            weights_name='20000_gmflow-animerun-v2-ft.safetensors', device=device
        ).eval()
    else:
        if unload:
            model_gmflow.to(device)

    if not isinstance(image1, torch.Tensor):
        image1 = img2tensor(image1, device=device)
    if not isinstance(image2, torch.Tensor):
        image2 = img2tensor(image2, device=device)

    input_h, input_w = image1.shape[-2:]
    if inference_size is not None:
        tgt_h = input_h
        tgt_w = input_w
        short_side = min(input_h, input_w)
        if short_side > inference_size:
            r = inference_size / short_side
            tgt_h = int(round(tgt_h * r))
            tgt_w = int(round(tgt_w * r))
        
        if tgt_h != input_h or tgt_w != input_w:
            image1 = torch.nn.functional.interpolate(image1, (tgt_h, tgt_w), mode='bilinear')
            image2 = torch.nn.functional.interpolate(image2, (tgt_h, tgt_w), mode='bilinear')

    padder = InputPadder(image1.shape, padding_factor=32)
    image1, image2 = padder.pad(image1, image2)


    # flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
    results_dict = model_gmflow(
                        image1, image2,
                        attn_splits_list=[2, 8],
                        corr_radius_list=[-1, 4],
                        prop_radius_list=[-1, 1],
                    )

    flow = padder.unpad(results_dict['flow_preds'][-1])

    oh, ow = flow.shape[-2:]
    if oh != input_h or ow != input_w:
        sh, sw = input_h / oh, input_w / ow
        flow = torch.nn.functional.interpolate(flow, (input_h, input_w), mode='bilinear')[0]
        scale = torch.tensor([sw, sh], dtype=flow.dtype, device=flow.device).reshape(-1, 2, 1, 1)
        flow = flow * scale
    
    if output_type == 'numpy':
        flow = tensor2img(flow.squeeze(), dtype=np.float32, clip=None)

    if unload:
        model_gmflow.to('cpu')

    return flow


model_raft: RAFT = None
@torch.inference_mode()
def apply_raft(image1, image2, device='cuda', unload=False, inference_size=384, output_type='numpy'):
    global model_raft
    if model_raft is None:
        def fix_raft_sd(model, state_dict):
            keys = list(state_dict.keys())
            for k in keys:
                p = state_dict.pop(k)
                state_dict[k.replace('module.', '')] = p
        args = Namespace(model=None, dataset=None, small=False, alternate_corr=False, mixed_precision=False)
        model_raft = init_model_from_pretrained(
            'dreMaz/AnimeRun_ckpt', RAFT, model_args={'args': args}, 
            weights_name='20000_raft-animerun-v2-ft_again.pth', device=device, patch_state_dict_func=fix_raft_sd
        ).eval()
    else:
        if unload:
            model_raft.to(device)

    if not isinstance(image1, torch.Tensor):
        image1 = img2tensor(image1, device=device)
    if not isinstance(image2, torch.Tensor):
        image2 = img2tensor(image2, device=device)

    input_h, input_w = image1.shape[-2:]
    if inference_size is not None:
        tgt_h = input_h
        tgt_w = input_w
        short_side = min(input_h, input_w)
        if short_side > inference_size:
            r = inference_size / short_side
            tgt_h = int(round(tgt_h * r))
            tgt_w = int(round(tgt_w * r))
        
        if tgt_h != input_h or tgt_w != input_w:
            image1 = torch.nn.functional.interpolate(image1, (tgt_h, tgt_w), mode='bilinear')
            image2 = torch.nn.functional.interpolate(image2, (tgt_h, tgt_w), mode='bilinear')

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    flow_low, flow_pr = model_raft(image1, image2, iters=32, test_mode=True)

    flow = padder.unpad(flow_pr[0]).cpu()

    oh, ow = flow.shape[-2:]
    if oh != input_h or ow != input_w:
        sh, sw = input_h / oh, input_w / ow
        flow = torch.nn.functional.interpolate(flow[None], (input_h, input_w), mode='bilinear')[0]
        scale = torch.tensor([sw, sh], dtype=flow.dtype, device=flow.device).reshape(-1, 2, 1, 1)
        flow = flow * scale
    
    if output_type == 'numpy':
        flow = tensor2img(flow.squeeze(), dtype=np.float32, clip=None)

    if unload:
        model_raft.to('cpu')

    return flow


def batch_apply_gmflow(frames, bsz=4, device='cuda', dtype = torch.float32):
    b, t, c, h, w = frames.shape
    first_frames = (frames[:, [0]] * 127.5 + 127.5).expand(-1, t - 1, -1, -1, -1).reshaspe(-1, c, h, w)
    sec_frames = (frames[:, 1:] * 127.5 + 127.5).view(-1, c, h, w)
    rst_list = []
    start_idx = 0
    total_bsz = first_frames.shape[0]
    while True:
        end_idx = min(start_idx + bsz, total_bsz)
        rst_list.append(
            apply_gmflow(first_frames[start_idx: end_idx], sec_frames[start_idx: end_idx], output_type='tensor')
        )
        if end_idx >= total_bsz:
            break
        start_idx += bsz
    rst_list = torch.cat(rst_list).reshape(b, t - 1, -1, h, w).to(device=device, dtype=dtype)
    return rst_list