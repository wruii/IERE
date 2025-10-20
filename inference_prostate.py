import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Subset
import albumentations as A
import torchmetrics
from tabulate import tabulate
import sys
sys.path.append('../')
from data.prostate_dataset import PretrainDataset
from torchmetrics.classification import Dice
import utils
import glob
import matplotlib
from medpy import metric
import pandas as pd
PRED_FOLDER = "inference"
all_centers = ['A-ISBI', 'B-ISBI_1.5', 'C-I2CVB', 'D-UCL', 'E-BIDMC', 'F-HK']

def out_of_domain_prediction(checkpoint_path: str,dataset_root='../../DeSAM-main/work_dir/train_embeddings/npz_files_vit_h',
                    output_folder=".", pad_to=224, center=0):
    from prostate.train_sam_prostate_cp import CardiacSegmentation
    model = CardiacSegmentation.load_from_checkpoint(checkpoint_path)
    model.eval()

    ious = []
    per_class_ious = []
    dice_list = []
    hd_list = []
    asd_list = []
    losses = []
    test_augs = []
    if pad_to is not None:
        test_augs.append(A.PadIfNeeded(pad_to, pad_to, border_mode=cv2.BORDER_CONSTANT, value=0, position='top_left'))
    transforms = A.Compose(test_augs)
    confusion_matrix = np.zeros((model.num_classes, model.num_classes))

    patientid_path = '../../DeSAM-main/work_dir/raw_data/prostate_patientid.csv'
    if not os.path.exists(patientid_path) and patientid_path.startswith('../../'):
        patientid_path = patientid_path[3:]
    if not os.path.exists(dataset_root) and dataset_root.startswith('../../'):
        dataset_root = dataset_root[3:]
    patientid = pd.read_csv(patientid_path)
    ood_centers = all_centers[center]

    all_patientid = patientid[patientid.center == ood_centers]['patientid'].values.tolist()
    if center == 1:
        used_patientid = all_patientid[int(len(all_patientid) * 0.7):]
        test_patientid = used_patientid[int(len(used_patientid) * 0.3):]
    else:
        test_patientid = all_patientid

    for patientid in tqdm(test_patientid):
        patientid_path_list = [x for x in os.listdir(dataset_root) if int(x[9:12]) == patientid]
        patientid_path_list = sorted(patientid_path_list)
        targets_numpy = []
        preds_numpy = []
        for patientid_path in patientid_path_list:
            npz_data = np.load(os.path.join(dataset_root, patientid_path), allow_pickle=True)
            image = npz_data['imgs'][:, :, 0] # original image
            gt = npz_data['gts']
            gt = cv2.resize(np.uint8(gt), (image.shape[0], image.shape[1]))
            gt[np.where(gt > 1)] = 1 # gt

            transformed = transforms(image=image, mask=gt)
            img, targets = transformed["image"], transformed["mask"]
            img = np.expand_dims(img, axis=0)
            targets = torch.from_numpy(targets.astype(np.int64))
            input_ = utils.pad_to_next_multiple_of_32(img)
            input_ = torch.from_numpy(input_)
            input_ = input_.unsqueeze(dim=0)
            _, logits, _ = model((input_.float()))
            if logits is not None:
                logits = logits[0, :, :img.shape[-2], :img.shape[-1]]  # Remove padding and the minibatch dim
                logits = logits.detach().cpu()
                preds = torch.argmax(logits, dim=0)

            iou = torchmetrics.functional.jaccard_index(preds, targets,
                                                        ignore_index=None, absent_score=1.0, num_classes=model.num_classes)
            ious.append(iou)
            per_class_ious.append(torchmetrics.functional.jaccard_index(preds, targets,
                                                                        absent_score=np.NaN, num_classes=model.num_classes,
                                                                        average='none'))
            confusion_matrix += torchmetrics.functional.confusion_matrix(preds, targets,
                                                                         num_classes=model.num_classes).numpy()

            if logits is not None:
                logits = logits[:image.shape[0], :image.shape[1]]
                loss = model.loss(logits.unsqueeze(dim=0), targets.unsqueeze(dim=0))
                losses.append(loss)
            else:
                losses.append(0)

            preds = preds.numpy()
            targets = targets.numpy()
            preds_numpy.append(preds)
            targets_numpy.append(targets)
            preds = preds[:image.shape[0], :image.shape[1]]
            plot_title = patientid_path + f"\nIoU: Mean = {iou:.3f}" + f" | prostate = {per_class_ious[-1][1]:.3f} "
            out_path = os.path.join(output_folder, os.path.basename(checkpoint_path).replace('ckpt', f'{center}'), patientid_path.replace(".npz", ".pdf"))
            utils.plot_acdc_prediction(image, preds, gt, plot_title, out_path)

        targets_numpy = np.array(targets_numpy)
        preds_numpy = np.array(preds_numpy)
        dice = metric.dc(np.uint8(preds_numpy), np.uint8(targets_numpy))
        hd = metric.binary.hd95(preds_numpy, targets_numpy)
        asd = metric.binary.asd(preds_numpy, targets_numpy)
        dice_list.append(dice)
        hd_list.append(hd)
        asd_list.append(asd)

    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(ious)
    mean_asd = np.mean(asd_list)
    mean_hd = np.mean(hd_list)
    std_iou = np.std(ious)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    per_class_ious_averaged_over_all_samples = np.nanmean(np.stack(per_class_ious), axis=0)
    np.save(os.path.join(output_folder, "per_class_ious"), np.stack(per_class_ious))
    print("=============================")
    print("Checkpoint", checkpoint_path)
    print(f"Evaluated {len(ious)} samples")
    print("Images padded to = ", pad_to)
    print("Mean Dice = ", mean_dice)
    print("Mean IoU =", mean_iou)
    print("Stdev IoU =", std_iou)
    print("Mean HD = ", mean_hd)
    print("Mean ASD =", mean_asd)
    print("Mean Loss =", mean_loss)
    print("Stdev Loss =", std_loss)
    print("Per-class-IoUs", per_class_ious_averaged_over_all_samples)
    print("Mean of per-class-IoUs", np.mean(per_class_ious_averaged_over_all_samples))
    print("Confusion matrix: \n", confusion_matrix)
    print("=============================")

    if not os.path.exists(os.path.join(output_folder, PRED_FOLDER)):
        os.makedirs(os.path.join(output_folder, PRED_FOLDER), exist_ok=True)
    labels = ["Background", "cancer"]
    utils.plot_iou_histograms(np.stack(per_class_ious), labels, os.path.join(output_folder, PRED_FOLDER))
    labels = ["bg", "cancer"]
    utils.plot_confmat(confusion_matrix, labels, os.path.join(output_folder, PRED_FOLDER))

    return mean_iou, std_iou, mean_dice, per_class_ious_averaged_over_all_samples, mean_loss, std_loss


def out_of_domain_prediction_ab(checkpoint_path: str,dataset_root='../../DeSAM-main/work_dir/train_embeddings/npz_files_vit_h',
                    output_folder=".", pad_to=224, center=0):
    from prostate.train_sam_prostate_cp_ab import CardiacSegmentation
    model = CardiacSegmentation.load_from_checkpoint(checkpoint_path)
    model.eval()

    ious = []
    per_class_ious = []
    dice_list = []
    hd_list = []
    asd_list = []
    losses = []
    test_augs = []
    if pad_to is not None:
        test_augs.append(A.PadIfNeeded(pad_to, pad_to, border_mode=cv2.BORDER_CONSTANT, value=0, position='top_left'))
    transforms = A.Compose(test_augs)
    confusion_matrix = np.zeros((model.num_classes, model.num_classes))

    patientid_path = '../../DeSAM-main/work_dir/raw_data/prostate_patientid.csv'
    if not os.path.exists(patientid_path) and patientid_path.startswith('../../'):
        patientid_path = patientid_path[3:]
    if not os.path.exists(dataset_root) and dataset_root.startswith('../../'):
        dataset_root = dataset_root[3:]
    patientid = pd.read_csv(patientid_path)
    ood_centers = all_centers[center]

    all_patientid = patientid[patientid.center == ood_centers]['patientid'].values.tolist()
    if center == 0:
        used_patientid = all_patientid[int(len(all_patientid) * 0.7):]
        test_patientid = used_patientid[int(len(used_patientid) * 0.3):]
    else:
        test_patientid = all_patientid

    for patientid in tqdm(test_patientid):
        patientid_path_list = [x for x in os.listdir(dataset_root) if int(x[9:12]) == patientid]
        patientid_path_list = sorted(patientid_path_list)
        targets_numpy = []
        preds_numpy = []
        for patientid_path in patientid_path_list:
            npz_data = np.load(os.path.join(dataset_root, patientid_path), allow_pickle=True)
            image = npz_data['imgs'][:, :, 0] # original image
            gt = npz_data['gts']
            gt = cv2.resize(np.uint8(gt), (image.shape[0], image.shape[1]))
            gt[np.where(gt > 1)] = 1 # gt

            transformed = transforms(image=image, mask=gt)
            img, targets = transformed["image"], transformed["mask"]
            img = np.expand_dims(img, axis=0)
            targets = torch.from_numpy(targets.astype(np.int64))
            input_ = utils.pad_to_next_multiple_of_32(img)
            input_ = torch.from_numpy(input_)
            input_ = input_.unsqueeze(dim=0)
            _, logits, _ = model((input_.float()))
            if logits is not None:
                logits = logits[0, :, :img.shape[-2], :img.shape[-1]]  # Remove padding and the minibatch dim
                logits = logits.detach().cpu()
                preds = torch.argmax(logits, dim=0)

            iou = torchmetrics.functional.jaccard_index(preds, targets,
                                                        ignore_index=None, absent_score=1.0, num_classes=model.num_classes)
            ious.append(iou)
            per_class_ious.append(torchmetrics.functional.jaccard_index(preds, targets,
                                                                        absent_score=np.NaN, num_classes=model.num_classes,
                                                                        average='none'))
            confusion_matrix += torchmetrics.functional.confusion_matrix(preds, targets,
                                                                         num_classes=model.num_classes).numpy()

            if logits is not None:
                logits = logits[:image.shape[0], :image.shape[1]]
                loss = model.loss(logits.unsqueeze(dim=0), targets.unsqueeze(dim=0))
                losses.append(loss)
            else:
                losses.append(0)

            preds = preds.numpy()
            targets = targets.numpy()
            preds_numpy.append(preds)
            targets_numpy.append(targets)
            preds = preds[:image.shape[0], :image.shape[1]]
            plot_title = patientid_path + f"\nIoU: Mean = {iou:.3f}" + f" | prostate = {per_class_ious[-1][1]:.3f} "
            out_path = os.path.join(output_folder, "checkpoints", os.path.basename(checkpoint_path).replace('.ckpt', f'_{center}'), patientid_path.replace(".npz", ".pdf"))
            utils.plot_acdc_prediction(image, preds, gt, plot_title, out_path)

        targets_numpy = np.array(targets_numpy)
        preds_numpy = np.array(preds_numpy)
        dice = metric.dc(np.uint8(preds_numpy), np.uint8(targets_numpy))
        hd = metric.binary.hd95(preds_numpy, targets_numpy)
        asd = metric.binary.asd(preds_numpy, targets_numpy)
        dice_list.append(dice)
        hd_list.append(hd)
        asd_list.append(asd)

    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(ious)
    mean_hd = np.mean(hd_list)
    mean_asd = np.mean(asd_list)
    std_iou = np.std(ious)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    per_class_ious_averaged_over_all_samples = np.nanmean(np.stack(per_class_ious), axis=0)
    np.save(os.path.join(output_folder, "per_class_ious"), np.stack(per_class_ious))
    print("=============================")
    print("Checkpoint", checkpoint_path)
    print(f"Evaluated {len(ious)} samples")
    print("Mean Dice = ", mean_dice)
    print("Mean HD = ", mean_hd)
    print("Mean ASD = ", mean_asd)
    print("Mean IoU =", mean_iou)
    print("Stdev IoU =", std_iou)

    return mean_iou, std_iou, mean_dice, per_class_ious_averaged_over_all_samples, mean_loss, std_loss

def multi_ckpt_eval(ckpt_paths, type='out', center=0):
    for ckpt_path in ckpt_paths:
        assert os.path.exists(ckpt_path), ckpt_path
    results = []
    for i in range(len(ckpt_paths)):
        ckpt_path = ckpt_paths[i]

        if type == 'out':#center_1_inference
            mean_iou, std_iou, mean_dice, per_class_ious, mean_loss, std_loss = out_of_domain_prediction(
                ckpt_path,
                output_folder=os.path.dirname(os.path.dirname(ckpt_path)),
                center=center
            )
        elif type == "out_ab":
            mean_iou, std_iou, mean_dice, per_class_ious, mean_loss, std_loss = out_of_domain_prediction_ab(
                ckpt_path,
                output_folder=os.path.dirname(os.path.dirname(ckpt_path)),
                center=center
            )

        ckpt_path_parts = ckpt_path.split(os.path.sep)
        checkpoint = ckpt_path_parts[-1]
        experiment = ckpt_path_parts[-4]
        ckpt_str = os.path.join(experiment, "...", checkpoint)
        ckpt_str = "`" + ckpt_str + "`"
        results.append([ckpt_str, mean_iou, std_iou, mean_dice, *per_class_ious])

    headers = ["Checkpoint", "Mean IoU", "Std IoU", "Mean Dice", "BG IoU", "DE IoU"]
    print(tabulate(results, headers, tablefmt="pipe"))


if __name__ == '__main__':
    ckpt_paths = glob.glob(os.path.join('checkpoints','*.ckpt'))

    # B TO B/A
    multi_ckpt_eval(ckpt_paths, type='out', center=1)
    multi_ckpt_eval(ckpt_paths, type='out', center=0)
    # A TO A/B
    multi_ckpt_eval(ckpt_paths, type='out_ab', center=0)
    multi_ckpt_eval(ckpt_paths, type='out_ab', center=1)
