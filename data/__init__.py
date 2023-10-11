'''
the codes for processing and loading data.
create by Xuying Zhang
'''

import torch.utils.data as data
import torch.distributed as dist
import torch
import torch.nn.functional as F


def get_dataloader(data_root, shot, batchsize=32, num_workers=8, mode='train'):

    #from data.refdataset_vae import R2CObjData as Dataset
    from data.refdataset import R2CObjData as Dataset
    if mode == 'train':
        print('load train data...')
        train_data = Dataset(
            data_root=data_root, 
            mode='train',
            shot=shot
        )
        Datasampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        train_loader = data.DataLoader(train_data, batch_size=batchsize, collate_fn=my_collate_fn, shuffle=True, num_workers=batchsize, pin_memory=True, sampler=None, drop_last=True)
        return Datasampler, train_loader

    elif mode == 'val' or mode == 'test':
        print('laod val data...')
        val_data = Dataset(
            data_root=data_root, 
            mode=mode,
            shot=shot,
        )

        val_loader = data.DataLoader(val_data, batch_size=1, collate_fn=my_collate_fn, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=None)
    
        return val_loader

    else:
        raise KeyError('mode {} error!'.format(mode))

def my_collate_fn(batchsize):
    size = 384
    imgs=[]
    labels=[]
    sal_f = []
    name = []
    for item in batchsize:
        imgs.append(F.interpolate(item[0], (size, size), mode='bilinear'))
        labels.append(F.interpolate(item[1], (size, size), mode='bilinear'))
        sal_f.append(item[2].unsqueeze(0))
        name.append(item[3])
    imgs = torch.cat(imgs, 0)
    labels = torch.cat(labels, 0)
    sal_f = torch.cat(sal_f, 0)
    # name = torch.cat(name, 0)
    return imgs, labels, sal_f, name
