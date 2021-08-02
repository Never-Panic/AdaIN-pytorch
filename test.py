import torch
from models import Network
import argparse
from pathlib import Path
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--style', type=str,
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
transform = T.Compose([
                T.ToTensor(),
            ])
content = transform(Image.open(args.content))
style = transform(Image.open(args.style))
style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)

# style transfer
with torch.inference_mode(): 
    out_image = network.style_transfer(content, style)

# save image
save_image(out_image, f'{args.output}/{content_dir.stem}_stylized_by_{style_dir.stem}.jpg')

