import torch.nn as nn
import torch.optim as optim
import argparse
import torch

from trainers.DogCatTrainer import DogCatTrainer
from models.DogCatModel import Resnet34

from metrics.classification_metric import ClassificationMetric
from utils.utils import set_seed

from dataset.DogCatDataset import CatDogDataset

from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomCrop, Normalize, Resize, RandomHorizontalFlip


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', help='Your training directory', default='data/train')
    parser.add_argument('--test_dir', help='Your testing directory', default='data/test')
    parser.add_argument('--image_size', help='Your training image size', default=224)
    parser.add_argument('--batch_size',help='Your training batch size',default=32)
    parser.add_argument('--num_workers', help='number of process', default=2)
    parser.add_argument('--seed',help='random seed',default=1234)
    parser.add_argument('--epoch', help='training epochs', default=5)
    parser.add_argument('--lr',help='learning rate',default=0.001)
    parser.add_argument('--val_batch_size', help='Your validation batch size', default=64)
    parser.add_argument('--grad_clip',help='gradient clipping theshold',default=5)
    parser.add_argument('--grad_accum_step', help='gradient accumalation step', default=1)
    parser.add_argument('--n_classses',help='Number of classes', default=2)
    parser.add_argument('--pretrained',help='Number of classes', default=1, type=bool)
    parser.add_argument('--gpu',help='Number of classes', default=1, type= bool)

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()

    train_transform = Compose([
        ToPILImage(),
        Resize(size=[args.image_size,args.image_size]),
        RandomCrop(size=[args.image_size, args.image_size]),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    model = Resnet34(args.n_classes,args.pretrained)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()
    metric = ClassificationMetric(n_classes=args.n_classes)
    train_dataset = CatDogDataset(file_path=args.file_dir,transform=train_transform)

    if args.gpu:
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    trainer = DogCatTrainer(model= model,
                        optimizer= optimizer,
                        criterion= criterion,
                        metric=metric,
                        log = None,
                        device = DEVICE
                        )

    trainer.train(train_dataset=train_dataset,
                  epochs=args.epoch,
                  gradient_accumalation_step=args.grad_accum_step,
                  train_batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  gradient_clipping=args.grad_clip)









