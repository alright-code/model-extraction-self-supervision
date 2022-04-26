from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor 

from pytorch_lightning import Trainer

from model import ImageNetClassifier
from dataset import ImageNetWithLogits


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    parser = ArgumentParser(description='ImageNet Testing', allow_abbrev=False)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--meta-dir', type=str, default='data')
    parser.add_argument('--logits-file', type=str, required=False)
    parser.add_argument('--chkpt', type=str, required=True)
    
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = ImageNetClassifier.load_from_checkpoint(args.chkpt)

    transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(mean=MEAN, std=STD)])
    test_data = ImageNetWithLogits(root=args.data_dir,
                                   logits_file=args.logits_file,
                                   split='test',
                                   meta_dir=args.data_dir,
                                   transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=32, num_workers=32, shuffle=False)

    trainer = Trainer.from_argparse_args(args)
    trainer.test(model, test_dataloader)


if __name__ == '__main__':
    main()