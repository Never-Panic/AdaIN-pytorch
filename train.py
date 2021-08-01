import argparse
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from models import Network
from dataset import getDataLoader

def adjust_learning_rate(optimizer, iteration_count):
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images',
                    default='/data/dataset/COCO2014')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images',
                    default='/data/dataset/WikiArt/images')
parser.add_argument('--vgg_path', type=str, default='./vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./checkpoints',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--save_every', type=int, default=10000)
parser.add_argument('--print_every', type=int, default=2000)
parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
args = parser.parse_args()

# basic setting
cudnn.benchmark = True

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

# network declare
network = Network(args.vgg_path)
network.train()
network.to(device)

if args.checkpoint_model:
        network.load_state_dict(torch.load(args.checkpoint_model))

# datasets
content_iter = iter(getDataLoader(args.content_dir, args.batch_size))
style_iter = iter(getDataLoader(args.style_dir, args.batch_size))

# optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

# training loop
for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)

    content_images = next(content_iter)[0].to(device)
    style_images = next(style_iter)[0].to(device)

    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if i % args.print_every==0:
        print(f'loss_content: {loss_c.item()}\n loss_style: {loss_s.item()}')

    if (i + 1) % args.save_every == 0 or (i + 1) == args.max_iter:
        torch.save(network.state_dict(), f"{args.save_dir}/network_checkpoint.pth")

writer.close()