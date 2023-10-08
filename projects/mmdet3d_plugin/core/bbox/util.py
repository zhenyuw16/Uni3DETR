import torch 
import numpy as np
import mmdet3d

__mmdet3d_version__ = float(mmdet3d.__version__[:3])


def normalize_bbox(bboxes, pc_range=None):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    # align coord system with previous version
    if __mmdet3d_version__ < 1.0:
        # w = bboxes[..., 3:4]
        # l = bboxes[..., 4:5]
        # h = bboxes[..., 5:6]
        w = bboxes[..., 3:4].log()
        l = bboxes[..., 4:5].log()
        h = bboxes[..., 5:6].log()
        rot = bboxes[..., 6:7]
    else:
        # l = bboxes[..., 3:4]
        # w = bboxes[..., 4:5]
        # h = bboxes[..., 5:6] 
        l = (bboxes[..., 3:4] + 1e-5).log()
        w = (bboxes[..., 4:5] + 1e-5).log()
        h = (bboxes[..., 5:6] + 1e-5).log()
        rot = bboxes[..., 6:7]
        rot = -rot - np.pi / 2
    
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range=None, version=0.8):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    
    # align coord system with previous version
    if __mmdet3d_version__ >= 1.0:
        rot = -rot - np.pi / 2
    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        if __mmdet3d_version__ < 1.0:
            denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
        else:
            denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot, vx, vy], dim=-1)
    else:
        if __mmdet3d_version__ < 1.0:
            denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
        else:
            denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot], dim=-1)
    return denormalized_bboxes

def bbox3d_mapping_back(bboxes, rot_degree, scale_factor, flip_horizontal, flip_vertical):
    """Map bboxes from testing scale to original image scale.

    Args:
        bboxes (:obj:`BaseInstance3DBoxes`): Boxes to be mapped back.
        scale_factor (float): Scale factor.
        flip_horizontal (bool): Whether to flip horizontally.
        flip_vertical (bool): Whether to flip vertically.

    Returns:
        :obj:`BaseInstance3DBoxes`: Boxes mapped back.
    """
    new_bboxes = bboxes.clone()
    if flip_horizontal:
        new_bboxes.flip('horizontal')
    if flip_vertical:
        new_bboxes.flip('vertical')
    new_bboxes.scale(1 / scale_factor)
    new_bboxes.rotate(-rot_degree)

    return new_bboxes

def get_rdiou(bboxes1, bboxes2):
    x1u, y1u, z1u = bboxes1[:,:,0], bboxes1[:,:,1], bboxes1[:,:,2]
    l1, w1, h1 =  torch.exp(bboxes1[:,:,3]), torch.exp(bboxes1[:,:,4]), torch.exp(bboxes1[:,:,5])
    t1 = torch.sin(bboxes1[:,:,6]) * torch.cos(bboxes2[:,:,6])
    x2u, y2u, z2u = bboxes2[:,:,0], bboxes2[:,:,1], bboxes2[:,:,2]
    l2, w2, h2 =  torch.exp(bboxes2[:,:,3]), torch.exp(bboxes2[:,:,4]), torch.exp(bboxes2[:,:,5])
    t2 = torch.cos(bboxes1[:,:,6]) * torch.sin(bboxes2[:,:,6])

    # we emperically scale the y/z to make their predictions more sensitive.
    x1 = x1u
    y1 = y1u * 2
    z1 = z1u * 2
    x2 = x2u
    y2 = y2u * 2
    z2 = z2u * 2

    # clamp is necessray to aviod inf.
    l1, w1, h1 = torch.clamp(l1, max=10), torch.clamp(w1, max=10), torch.clamp(h1, max=10)
    j1, j2 = torch.ones_like(h2), torch.ones_like(h2)

    volume_1 = l1 * w1 * h1 * j1
    volume_2 = l2 * w2 * h2 * j2

    inter_l = torch.max(x1 - l1 / 2, x2 - l2 / 2)
    inter_r = torch.min(x1 + l1 / 2, x2 + l2 / 2)
    inter_t = torch.max(y1 - w1 / 2, y2 - w2 / 2)
    inter_b = torch.min(y1 + w1 / 2, y2 + w2 / 2)
    inter_u = torch.max(z1 - h1 / 2, z2 - h2 / 2)
    inter_d = torch.min(z1 + h1 / 2, z2 + h2 / 2)
    inter_m = torch.max(t1 - j1 / 2, t2 - j2 / 2)
    inter_n = torch.min(t1 + j1 / 2, t2 + j2 / 2)

    inter_volume = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0) \
        * torch.clamp((inter_d - inter_u),min=0) * torch.clamp((inter_n - inter_m),min=0)
    
    c_l = torch.min(x1 - l1 / 2,x2 - l2 / 2)
    c_r = torch.max(x1 + l1 / 2,x2 + l2 / 2)
    c_t = torch.min(y1 - w1 / 2,y2 - w2 / 2)
    c_b = torch.max(y1 + w1 / 2,y2 + w2 / 2)
    c_u = torch.min(z1 - h1 / 2,z2 - h2 / 2)
    c_d = torch.max(z1 + h1 / 2,z2 + h2 / 2)
    c_m = torch.min(t1 - j1 / 2,t2 - j2 / 2)
    c_n = torch.max(t1 + j1 / 2,t2 + j2 / 2)

    inter_diag = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (t2 - t1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2 + torch.clamp((c_d - c_u),min=0)**2  + torch.clamp((c_n - c_m),min=0)**2

    union = volume_1 + volume_2 - inter_volume
    u = (inter_diag) / c_diag
    rdiou = inter_volume / union
    return u, rdiou