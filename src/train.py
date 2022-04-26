from argparse import ArgumentParser

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

from dataset import ImageNetWithLogits
from model import ImageNetClassifier

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    parser = ArgumentParser(description="ImageNet Training", allow_abbrev=False)
    parser.add_argument(
        "--data-dir", type=str, default="data", help="ImageNet data directory."
    )
    parser.add_argument(
        "--logits-file",
        type=str,
        required=False,
        help="Extracted target logits file path.",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Percentage of labels to finetune with.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=32)

    # Model
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument(
        "--pretrained-chkpt",
        required=False,
        type=str,
        help="Self-supervised pretrained model path (resnet50).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5, help="Distillation temperature."
    )

    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = ImageNetClassifier(**args.__dict__)

    if args.finetune:
        transform = Compose(
            [
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        transform = Compose(
            [
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ColorJitter(0.3, 0.3, 0.3, 0.1),
                ToTensor(),
                Normalize(mean=MEAN, std=STD),
            ]
        )
    train_data = ImageNetWithLogits(
        root=args.data_dir,
        logits_file=args.logits_file,
        data_fraction=args.data_fraction,
        split="train",
        meta_dir=args.data_dir,
        num_imgs_per_class_val_split=0,
        transform=transform,
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()
