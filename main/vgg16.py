import numpy as np
import torch.nn as nn
import torch
from torch.utils import data
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import os
import sys
import json
from torch.nn import init
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from AddGaussianNoise import AddGaussianNoise
import yaml
from yaml.loader import SafeLoader


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]

    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16_(nn.Module):
    def __init__(self, n_classes=1000, init_weights: bool = True):
        super(VGG16_, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([1, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        # self.layer6 = vgg_fc_layer(7*7*512, 4096) # when 224
        self.layer6 = vgg_fc_layer(3 * 3 * 512, 4096)  # when input 128
        # self.layer6 = vgg_fc_layer(3*8192, 4096)
        # self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        vgg16_features = self.layer5(out)
        # print(str(vgg16_features.size())+"features shape")
        out = vgg16_features.view(vgg16_features.size(0), -1)
        # print(str(out.size())+"view-1")
        out = self.layer6(out)
        # out = self.layer7(out)
        out = self.layer8(out)

        return out

    def _initialize_weights(self):
        # 初始化参数，是卷积层的话，使用凯明初始化，BN层的话，我们使用常数初始化，Linear层的话，我们使用高斯初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def main(cfg: dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # transforms.RandomApply(AddGaussianNoise(args.mean, args.std), p=0.5)
    data_transform = {
        "train": transforms.Compose([transforms.Resize((96, 96)),
                                     transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor(),
                                     #  AddGaussianNoise(0.001, 0.009),
                                     #  AddGaussianNoise(args.mean, args.std)
                                     transforms.RandomApply([AddGaussianNoise(0.001, 0.001)], p=0.01),
                                     transforms.Normalize(0.883, 0.201)
                                     #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]),
        "val": transforms.Compose([transforms.Resize((96, 96)),
                                   transforms.ToTensor(),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.Normalize(0.883, 0.201)
                                   #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])
    }

    train_dataset = torchvision.datasets.ImageFolder(cfg['traindir'], transform=data_transform["train"])

    train_num = len(train_dataset)

    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    # json_str = json.dumps(cla_dict, indent=300)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    save_path = cfg["save_path"]
    batch_size = cfg["batch_size"]
    nw = cfg["nw"]
    epochs = cfg["epochs"]
    best_acc = 0.0
    # net = handVGG(301)
    # net = torchvision.models.vgg16_bn()
    net = VGG16_(n_classes=cfg["num_classes"])
    net.to(device)

    # lr = 0.0002 intial lr
    lr = cfg["lr"]
    # lambda1 = lambda : 0.90
    print("Use " + str(net) + "epochs " + str(epochs) + "classes:" + str(cfg["num_classes"]) + "epochs: " + str(
        cfg["epochs"]) + "lr: " + str(cfg["lr"]))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    # scheduler = lr_scheduler(optimizer, lambda1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer,lambda1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    loss_function = nn.CrossEntropyLoss()

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size, shuffle=True,
    #                                            num_workers=nw)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(cfg["testdir"],
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    train_num = len(train_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        # scheduler.step()
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            # logits, aux_logits2, aux_logits1 = net(images.to(device))
            logits = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            # loss1 = loss_function(aux_logits1, labels.to(device))
            # loss2 = loss_function(aux_logits2, labels.to(device))
            # loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss = loss0
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # scheduler.step()
        # validate

        scheduler.step()
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        acctrain = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        train_acc = 0.0
        if ((epoch + 1) % 5 == 0):
            with torch.no_grad():

                val_bar1 = tqdm(train_loader, file=sys.stdout)
                for _ in val_bar1:
                    _img, _labels = _
                    outs = net(_img.to(device))
                    _predict = torch.max(outs, dim=1)[1]
                    acctrain += torch.eq(_predict, _labels.to(device)).sum().item()
                train_acc = acctrain / train_num
                print('train_acc: %.6f' % train_acc)

        val_accurate = acc / val_num

        print('[epoch %d] train_loss: %.6f,  val_accuracy: %.6f ' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    data = []
    with open('../cfg.yml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(["traindir"])
    main(data)
