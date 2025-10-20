import os
import glob
import shutil
from tqdm import tqdm
import pydoc
import omegaconf
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import contextlib
import pdb


def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_pixel_x, patch_pixel_y = int(img_x*mask_ratio), int(img_y*mask_ratio)
    w = np.random.randint(0, 224 - patch_pixel_x)
    h = np.random.randint(0, 224 - patch_pixel_y)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y] = 0
    return mask.long(), loss_mask.long()


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['mask']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'mask': label}

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')

def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def flatten_dict(d, top_level_key="", sep="_"):
    flat_d = {}
    for k, v in d.items():
        if isinstance(v, (dict, omegaconf.DictConfig)):
            flat_d.update(flatten_dict (v, top_level_key=(k + sep)))
        else:
            flat_d[top_level_key + k] = v
    return flat_d

def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs) 

    return pydoc.locate(object_type)(**kwargs)
    

def merge_pdfs(base_path=".", individual_pdfs_folder="", pdf_path_list=None, out_file_name = "merged.pdf", remove_files=False):
    from PyPDF2 import PdfFileMerger

    if pdf_path_list is None:
        inputs_folder = os.path.join(base_path, individual_pdfs_folder)
        pdf_paths = glob.glob(os.path.join(inputs_folder, "*.pdf"), recursive=True)
        if not pdf_paths: 
            # No pdfs to merge were found
            return
    else:
        inputs_folder = os.path.commonpath(pdf_path_list)
        pdf_paths = pdf_path_list
    
    merger = PdfFileMerger()

    for pdf in tqdm(pdf_paths):
        merger.append(pdf)
        if remove_files:
            os.remove(pdf)

    merged_pdf_path = os.path.join(inputs_folder, out_file_name)
    merger.write(merged_pdf_path)
    merger.close()
    print("Merged pdf saved to: ", os.path.abspath(merged_pdf_path))

    if remove_files:
        print("Removing folder", inputs_folder)
        shutil.rmtree(inputs_folder)

def mask2box( mask):
    h, w = mask.shape
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    # 解析左上角行列号
    left_top_r = np.min(rows)  # y
    left_top_c = np.min(clos)  # x
    # 解析右下角行列号
    right_bottom_r = np.max(rows)  # y
    right_bottom_c = np.max(clos)  # x
    # # 转换为[x,y,w,h]
    x = left_top_c
    y = left_top_r
    assert x >= 0, y >= 0
    w = right_bottom_c - left_top_c
    h = right_bottom_r - left_top_r
    if w == 0:
        w += 1
    if h == 0:
        h += 1
    assert w > 0, h > 0
    return [x, y, w, h]

