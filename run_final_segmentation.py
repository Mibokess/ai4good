import torch
from PIL import Image
import numpy as np
import os, glob
from systems.seg_system_jaccard import SegmentationSystem
from torchvision import transforms
import segmentation_models_pytorch as smp
from configs.final_seg_config import command_line_parser

def main():
    cfg = command_line_parser()

    data_dir = cfg.dataset_root
    save_dir = cfg.save_root
    seg_checkpoint_path = cfg.segnet_checkpoint

    color_img_paths = glob.glob(os.path.join(data_dir,"*.png"))
    transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])
    net = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    model = SegmentationSystem.load_from_checkpoint(seg_checkpoint_path, net=net)
    net = model.net
    for img_path in color_img_paths:
        save_name = save_dir + '/' + img_path.rsplit('/', 1)[-1] 
        color_img = Image.open(img_path)
        color_img = transform(color_img)
        color_img = torch.unsqueeze(color_img, dim=0)
        seg_img = net(color_img)
        seg_img = torch.squeeze(seg_img,dim=0)
        seg_img = torch.clamp(seg_img, min=0.0, max=1.0) 
        seg_img = seg_img.int().numpy()
        seg_img = seg_img[0]
        rescaled = (255.0/seg_img.max() * (seg_img - seg_img.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(save_name)

if __name__ == "__main__":
    main()