from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, ColorJitter, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor 

from pytorch_lightning import Trainer

from model import ImageNetClassifier
from dataset import ImageNetWithLogits


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    parser = ArgumentParser(description='ImageNet Training', allow_abbrev=False)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--logits-file', type=str, required=False)
    parser.add_argument('--data-fraction', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=256)

    # Model
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--pretrained-chkpt', required=False, type=str)
    parser.add_argument('--temperature', type=float, default=1.5)
    
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = ImageNetClassifier(**args.__dict__)

    if args.finetune:
        transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean=MEAN, std=STD)])
    else:
        transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ColorJitter(0.3, 0.3, 0.3, 0.1), ToTensor(), Normalize(mean=MEAN, std=STD)])
    train_data = ImageNetWithLogits(root=args.data_dir,
                                    logits_file=args.logits_file,
                                    data_fraction=args.data_fraction,
                                    split='train',
                                    meta_dir=args.data_dir,
                                    num_imgs_per_class_val_split=0,
                                    transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=32, shuffle=True)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    main()