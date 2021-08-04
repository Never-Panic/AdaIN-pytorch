import torch
from torchvision.transforms.transforms import ToTensor
from models import Network
import argparse
from pathlib import Path
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image

'''
run command:
CUDA_VISIBLE_DEVICES=7 python3 test.py --checkpoint ./checkpoints/
'''

parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str,
                    default='./content/tele.jpg',
                    help='File path to the content image')
parser.add_argument('--style', type=str,
                    default='./style/starry_night.jpg',
                    help='File path to the style image')
parser.add_argument('--vgg_path', type=str, default='./vgg_normalised.pth')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/network_checkpoint.pth')
parser.add_argument('--output', type=str, default='./output',
                    help='Directory to save the output image(s)')
args = parser.parse_args()

# basic setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)
content_dir = Path(args.content)
style_dir = Path(args.style)

# network declare
network = Network(args.vgg_path)
network.load_state_dict(torch.load(args.checkpoint))
network.eval()
network.to(device)

# load image
transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(700),
    ]
)
content = transform(Image.open(args.content))
style = transform(Image.open(args.style))

content = content.to(device).unsqueeze(0)
style = style.to(device).unsqueeze(0)

# style transfer
with torch.no_grad(): 
    out_image = network.style_transfer(content, style)

# save image
content = TF.resize(content, out_image.size()[-2:])
style = TF.resize(style, out_image.size()[-2:])
save_image([content.squeeze(), out_image.squeeze(), style.squeeze()], f'{args.output}/{content_dir.stem}_stylized_by_{style_dir.stem}.jpg')

