import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from misc import utils

def default_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    return RGBimg

def RGBHSV_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    return RGBimg, HSVimg

def DepthImg_loader(path,imgsize=32):
    img = Image.open(path)
    re_img = img.resize((imgsize, imgsize), resample=Image.BICUBIC)
    return re_img


class DatasetLoader(Dataset):
    def __init__(self, args, name, transform=None, 
                loader=default_loader, depthimg_loader=DepthImg_loader, rgbhsv_loader = RGBHSV_loader,
                root='../../../datasets/'):
        self.args = args
        self.name = name
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, self.name)
        filename = 'image_list_all.txt'

        fh = open(os.path.join(self.root, filename), 'r')

        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()

            dirlist = words[0].strip().split('/')
            imgname = dirlist[-1][:-4]
            
            if int(words[1])==1 and name=='idiap':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], dirlist[2], imgname + '_depth.jpg')
            elif int(words[1])==1 and name=='CASIA':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], dirlist[2], imgname + '_depth.jpg')
            elif int(words[1])==1 and name=='MSU':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], imgname + '_depth.jpg')  
            elif int(words[1])==1 and name=='OULU':
                depth_dir = os.path.join('depth', dirlist[0], imgname + '_depth.jpg')
            elif int(words[1])==1 and name=='SiW':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], dirlist[2], dirlist[3],imgname + '_depth.jpg')                                
            else:
                depth_dir = os.path.join('depth', 'fake_depth.jpg') 

            imgs.append((words[0], depth_dir, int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.depthimg_loader = depthimg_loader
        self.rgbhsv_loader = rgbhsv_loader

    def __getitem__(self, index):
        fn, depth_img_dir, label = self.imgs[index]
        fn = os.path.join(self.root, fn)
        depth_img_dir_all = os.path.join(self.root, depth_img_dir)

        if self.args.net_type is 'DepthAux':
            rgbimg, hsvimg = self.rgbhsv_loader(fn)
        else:
            rgbimg = self.loader(fn)
        
        depthimg = self.depthimg_loader(depth_img_dir_all)

        if self.transform is not None:
            if self.args.net_type is 'DepthAux':
                rgbimg = self.transform(rgbimg)
                hsvimg = self.transform(hsvimg)
                img = torch.cat([rgbimg,hsvimg],0)
            else:
                img = self.transform(rgbimg)

            depth_transforms = transforms.ToTensor()
            depthimg = depth_transforms(depthimg)

        return img, depthimg, label

    def __len__(self):
        return len(self.imgs)


def get_tgtdataset_loader(args, name, batch_size):

    if args.net_type is 'Resnet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        te_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            normalize]) 

    elif args.net_type in ['CDCN', 'DepthAux'] :
        te_transforms = transforms.Compose([transforms.ToTensor()])

    # dataset and data loader
    dataset = DatasetLoader(args, name=name,
                        transform=te_transforms
                        )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=True)

    return data_loader
