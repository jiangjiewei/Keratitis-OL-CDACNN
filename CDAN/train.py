import os
import math
import argparse

import torch
import torch.optim as optim
from torchvision import transforms

from model_DA import densenet121_DA, load_state_dict
from model import densenet121
from cost_sensitive import  train_one_epoch, evaluate

from torchvision import transforms, datasets
import json
from torchtoolbox.transform import Cutout


def main(args):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if args.data == "conj-c":
        mean = [0.5841176, 0.35896605, 0.2481438]
        std = [0.20916606, 0.18940377, 0.17913769]
    elif args.data =="c":
        mean = [0.55412745, 0.34579736, 0.23483945]
        std = [0.19785675, 0.17708106, 0.17046732]
    else:
        mean = [0.5765036, 0.34929818, 0.2401832]
        std = [0.2179051, 0.19200659, 0.17808074]

    data_transform = {

        "train": transforms.Compose([transforms.Resize([224,224]),
                                     Cutout(),
                                     transforms.RandomRotation(90),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean,std)]),
        "val": transforms.Compose([transforms.Resize([224,224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean,std)])}

    data_root = os.path.abspath( "../data")  # get data root path
    if args.data == "conj-c":
        image_path = os.path.join(data_root,  "conjunctival-corneal")
    elif args.data =="c":
        image_path = os.path.join(data_root,  "corneal")
    else:
        image_path = os.path.join(data_root,  "original")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)  

    classes_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)


    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])


    val_num = len(validate_dataset)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    if args.attention:
        model = densenet121_DA(num_classes=args.num_classes).to(device)
    else:
        model = densenet121(num_classes=args.num_classes).to(device)
    if os.path.exists(args.weights):
        load_state_dict(model, args.weights)
    print(model)


    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)
                print('freeze layer')

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg,lr=args.lr)#lr=0.0001
    weight_dir = './weights'
    if os.path.exists(weight_dir)==False:
        os.mkdir(weight_dir)
        print('weights successfully created')

    best_acc = 0
    for epoch in range(args.epochs):

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)


        acc = evaluate(model=model,
                           data_loader=validate_loader,
                           device=device)


        #
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./weights/best_model.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--weights', type=str, default='densenet121.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--attention', action = 'store_true')

    parser.add_argument('--data', type=str, default='conj-c')


    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
