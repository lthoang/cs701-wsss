import os
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib


# from voc12 import dataloader
from cs701 import dataloader
from misc import pyutils, torchutils


def export_prediction(model, data_loader, save_path='./label.txt'):
    print('exporting prediction labels to file {} ... '.format(save_path), flush=True, end='')
    model.eval()

    result = {}
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img'].cuda(non_blocking=True)
            filenames = pack['name']
            x = model(img)
            preds = (torch.sigmoid(x) > 0.5).to(torch.float32)
            for filename, pred in zip(filenames, preds):
                result[filename] = pred.nonzero().flatten().tolist()
                if len(result[filename]) == 0:
                    result[filename] = [0] # set this label by default

    result = dict(sorted(result.items()))
    with open(save_path, 'w') as f:
        for fname, plabels in result.items():
            f.write('{} {}\n'.format(fname, ' '.join([str(lb) for lb in plabels])))
    return




def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model = torch.nn.DataParallel(model).cuda()
    test_dataset = dataloader.ImageDataset(args.infer_list, dataset_root=args.dataset_root, 
                                           crop_size=512)
    test_data_loader = DataLoader(test_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    export_prediction(model, test_data_loader, args.output_label_path)
