import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from misc import utils

def OriImg_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    return RGBimg

def RGBHSV_loader(path, resize=False):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    if resize:
        RGBimg = RGBimg.resize((128, 128), resample=Image.BICUBIC)
        HSVimg = HSVimg.resize((128, 128), resample=Image.BICUBIC)

    return RGBimg, HSVimg

def DepthImg_loader(path,imgsize=32):
    img = Image.open(path)
    re_img = img.resize((imgsize, imgsize), resample=Image.BICUBIC)
    return re_img

class DatasetLoader(Dataset):
    def __init__(self, args, name, getreal, transform, 
                    oriimg_loader=OriImg_loader, depthimg_loader=DepthImg_loader, rgbhsv_loader = RGBHSV_loader,
                    attacktype='all', root='../../../datasets/'):

        self.args = args
        self.name = name
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, self.name)
        if getreal:
            filename = 'image_list_real.txt'
        else:
            print('attacktype: {}'.format(attacktype))
            if attacktype is 'print':
                filename = 'image_list_print.txt' 
            elif attacktype is 'video':
                filename = 'image_list_video.txt'
            elif attacktype is 'all':
                filename = 'image_list_fake.txt'

        fh = open(os.path.join(self.root, filename), 'r')

        imgs = []
        for line in fh:

            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()

            dirlist = words[0].strip().split('/')
            imgname = dirlist[-1][:-4]

            if getreal and name=='idiap':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], dirlist[2], imgname + '_depth.jpg')
            elif getreal and name=='CASIA':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], dirlist[2], imgname + '_depth.jpg')
            elif getreal and name=='MSU':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], imgname + '_depth.jpg')  
            elif getreal and name=='OULU':
                depth_dir = os.path.join('depth', dirlist[0], imgname + '_depth.jpg')
            elif getreal and name=='SiW':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], dirlist[2], dirlist[3],imgname + '_depth.jpg')
            elif getreal and name=='3DMAD':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], imgname + '_depth.jpg')
            elif getreal and name=='HKBUMARsV2':
                depth_dir = os.path.join('depth', dirlist[0], dirlist[1], imgname + '_depth.jpg')                                      
            else:
                depth_dir = os.path.join('depth', 'fake_depth.jpg') 

            imgs.append((words[0], depth_dir, int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.oriimg_loader = oriimg_loader
        self.depthimg_loader = depthimg_loader
        self.rgbhsv_loader = rgbhsv_loader

    def __getitem__(self, index):
        ori_img_dir, depth_img_dir, label = self.imgs[index]
        ori_img_dir_all = os.path.join(self.root, ori_img_dir)
        depth_img_dir_all = os.path.join(self.root, depth_img_dir)

        if self.args.net_type is 'DepthAux':
            rgbimg, hsvimg = self.rgbhsv_loader(ori_img_dir_all)
            rgbimg_resize, hsvimg_resize = self.rgbhsv_loader(ori_img_dir_all, resize=False)
        else:
            img = self.oriimg_loader(ori_img_dir_all)

        depth_img = self.depthimg_loader(depth_img_dir_all)

        if self.transform is not None:
            if self.args.net_type is 'DepthAux':
                rgbimg = self.transform(rgbimg)
                hsvimg = self.transform(hsvimg)
                img = torch.cat([rgbimg,hsvimg],0)

                rgbimg_resize = self.transform(rgbimg_resize)
                hsvimg_resize = self.transform(hsvimg_resize)
                img_resize = torch.cat([rgbimg_resize,hsvimg_resize],0)
            else:
                img = self.transform(img)

            depth_transforms = transforms.ToTensor()
            depth_img = depth_transforms(depth_img)
        
        return img, depth_img, label, img_resize

    def __len__(self):
        return len(self.imgs)


def get_dataset_loader(args, name, getreal, batch_size, attacktype='all'):
    
    if args.net_type is 'Resnet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([])

        train_transform.transforms.append(transforms.RandomResizedCrop(224))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)

    elif args.net_type in ['CDCN', 'DepthAux'] :
        train_transform = transforms.Compose([transforms.ToTensor()])  

    # dataset and data loader
    dataset = DatasetLoader(args=args, name=name,
                        getreal=getreal,
                        attacktype = attacktype,
                        transform=train_transform
                        )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True)

    return data_loader
