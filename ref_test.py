import torch
import FSPNet_model
import dataset
import os
from data import get_dataloader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite

if __name__ =='__main__':
    batch_size = 1
    net = FSPNet_model.Model(None, img_size=384).cuda()

    ckpt=['model_36_loss_0.24612.pth']

    Dirs=["./Datasets/R2C7K/"]

    result_save_root="./path_to_save_root/results"

    for m in ckpt:
        print(m)
        # pretrained_dict = torch.load("./ckpt/"+m)['model']
        ckpt_root="./ref_fspnet/"
        ckpt_file="ckpt_save/"
        pretrained_dict = torch.load(ckpt_root+ckpt_file+m)

        net_dict = net.state_dict()
        pretrained_dict={k[7:]: v for k, v in pretrained_dict.items() if k[7:] in net_dict }
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        net.eval()
        for i in range(len(Dirs)):
            Dir = Dirs[i]
            if not os.path.exists(result_save_root):
                os.makedirs(result_save_root)
            if not os.path.exists(os.path.join(result_save_root, Dir.split("/")[-1])):
                os.makedirs(os.path.join(result_save_root, Dir.split("/")[-1]))
            # Dataset = dataset.TestDataset(Dir, 384)
            # Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size*2)
            Dataloader = get_dataloader(Dir, 5, 352, batchsize=batch_size, num_workers=batch_size*2, mode='test')
            count=0
            for data in Dataloader:
                count+=1
                cuda_device = torch.device('cuda')
                img, label, supp_feat, name = data[0].to(cuda_device), data[1].to(cuda_device), data[2].to(cuda_device), data[3]
                name = name[0].split("/")[-1]
                with torch.no_grad():
                    out = net(img, supp_feat)[3]
                    # out = net(img)
                B,C,H,W = label.size()
                o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
                o =(o-o.min())/(o.max()-o.min()+1e-8)
                o = (o*255).astype(np.uint8)
                imwrite(result_save_root+Dir.split("/")[-1]+"/"+name+'.png', o)
    
    print("Test finished!")


