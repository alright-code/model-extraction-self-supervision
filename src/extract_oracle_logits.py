from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    parser = ArgumentParser(description="Oracle")
    parser.add_argument(
        "--torch-hub-dir",
        type=str,
        default="torchhub",
        help="Directory to download the oracle model to.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/", help="Imagenet data directory."
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="oracle-logits.pt",
        help="Output file for the oracle logit predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=64)

    args = parser.parse_args()

    # Standard imagenet image transformation to a 3x224x224 input.
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    print("Loading ImageNet...")
    dataset = ImageNet(args.data_dir, transform=transform)
    num_images = len(dataset)
    dataloader = DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False
    )

    print("Loading Model...")
    torch.hub.set_dir(args.torch_hub_dir)
    model = torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnext101_32x16d_swsl"
    )
    model.cuda()
    model.eval()

    print("Extracting logits...")
    output_logits = torch.empty((num_images, 1000))
    idx = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.cuda()

            logits = model(x)
            preds = logits.argmax(1)

            logits = logits.cpu()

            output_logits[idx : (idx + len(logits))] = logits

            idx += len(logits)

    print("Saving Logits...")
    torch.save(output_logits, args.out_file)

    print("Done!")


if __name__ == "__main__":
    main()
