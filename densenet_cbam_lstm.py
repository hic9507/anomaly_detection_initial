# 임포트 import
import os
import numpy as np
from tqdm import tqdm
import glob
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, dataset, DataLoader, RandomSampler
from collections import OrderedDict
from torchvision.models import densenet121

from sklearn.metrics import f1_score, accuracy_score, auc, roc_auc_score, roc_curve

from torch.nn.modules import pooling
from torch.nn.modules.flatten import Flatten
from torch.utils import data
from sklearn.metrics import f1_score, accuracy_score

args = {"data_folder": "D:\\abnormal_detection_dataset\\test_data\\",
        "graphs_folder": "./graph_I3D/UBI-FIGHTS-VIDEO/", "epoch": 2, "batch_size": 1, "num_class": 1, "learning_rate": 0.001,  # 원래 1e-4
        "decay_rate": 0.998, "num_workers": 4, "img_size": (640, 360), "img_depth": 3, "FPS": 150} # decay_rate 원래 0.98 "img_size": (320, 240)

# gpu 설정
device = torch.device('cuda:0')

# 그래프 plot 폴더 경로 설정
if not os.path.exists(args["graphs_folder"]) : os.mkdir(args["graphs_folder"])
model_save_folder = './trained_model_I3D/UBI-FIGHTS-VIDEO/'

if not os.path.exists(model_save_folder) : os.mkdir(model_save_folder)


##### CBAM 정의  Channel_Attnetion > ChannelPool > Spatial_Attention > CBAM 순으로 정의
class Channel_Attention(nn.Module):

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types
        self.shared_mlp = nn.Sequential(nn.Flatten(), nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),
                                        nn.ReLU(inplace=True), nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in))

    def forward(self, x):
        channel_attentions = []
        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))
        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scaled

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()
        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1,
                                                         padding=(kernel_size-1)//2, bias=False), nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True))

    def forward(self, x):
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        scaled = nn.Sigmoid()(x_output)
        return x * scaled

class CBAM(nn.Module):
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)
        self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        x_out = self.spatial_attention(x_out)
        return x_out

# DenseNet BottleNeck
class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4 * growth_rate
        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(), nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels), nn.ReLU(), nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False))
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat([self.shortcut(x), self.residual(x)], 1)

# Transition Block: reduce feature map size and number of channels
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False), nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        return self.down_sample(x)

# DenseNet
class DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate=12, reduction=0.5, num_classes=args["num_class"], init_weights=True):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate  # output channels of conv1 before entering Dense Block
        self.conv1 = nn.Sequential(nn.Conv2d(3, inner_channels, 7, stride=2, padding=3), nn.MaxPool2d(3, 2, padding=1))
        self.features = nn.Sequential()

        for i in range(len(nblocks) - 1):
            self.features.add_module('dense_block_{}'.format(i), self._make_dense_block(nblocks[i], inner_channels))
            inner_channels += growth_rate * nblocks[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(i), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module('dense_block_{}'.format(len(nblocks) - 1),
                                 self._make_dense_block(nblocks[len(nblocks) - 1], inner_channels))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(264, inner_channels,batch_first=True)
        self.linear = nn.Linear(inner_channels, num_classes)

        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_dense_block(self, nblock, inner_channels):
        dense_block = nn.Sequential()
        for i in range(nblock):
            dense_block.add_module('bottle_neck_layer_{}'.format(i), BottleNeck(inner_channels, self.growth_rate))
            inner_channels += self.growth_rate
        return dense_block

    def _initialize_weights(self):
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

def DenseNet_121():
    return DenseNet([6, 12, 24, 6])

###### CustomDataset 정의
class CustomDataset(Dataset):

    def __init__(self, dataset_folder_path=args["data_folder"], image_size=args["img_size"], image_depth=3, train=True, transform=None):
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()

    def __len__(self):
        return len(self.image_path_label)

    def __getitem__(self, idx):
        frames = []
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, label = self.image_path_label[idx]
        cap = cv2.VideoCapture(image)

        for frame in range(args["FPS"]):
            frame, image = cap.read()
            if self.image_depth == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255
                frames.append(image)
            else:
                image = image / 255
                frames.append(image)
        return {'image': torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2), 'label': label}

    def get_classnames(self):
        return os.listdir(f"{self.dataset_folder_path}train/")

    def read_folder(self):
        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path}train/"
        else:
            folder_path = f"{self.dataset_folder_path}val/"  # valid도 가능

        for x in glob.glob(folder_path + "**", recursive=True):  # 해당 폴더의 하위 폴더까지 탐색
            if not x.endswith('mp4'):
                continue
            class_idx = self.classes.index(x.split('\\')[-2])
            image_path_label.append((x, int(class_idx)))
        return image_path_label

##### 콜백함수
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path=f"{model_save_folder}UBI-FIGHTS_640x360_videocapture_model_early_densenet_cbam_lstm.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print('--------------------------------------------------------------------------------\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  val loss is decreased, Saving model ...')
            print('\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

##### Train
def train(args, device="cuda:0"):
    model = DenseNet_121()
    model.to(device)

    optimizer = Adam(model.parameters(), args["learning_rate"])
    lr_decay = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    ##### train, test dataset 정의
    train_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"], image_depth=args["img_depth"], train=True, transform=transforms.ToTensor())
    test_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"], image_depth=args["img_depth"], train=False, transform=transforms.ToTensor())

    train_generator = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"], pin_memory=True)
    test_generator = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"], pin_memory=True)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    best_accuracy = 0
    valid_losses = []
    y_true = []
    y_pred = []

    early_stopping = EarlyStopping(verbose=True)

    for epoch in range(args["epoch"]):
        model.train()

        loss_per_epoch = []
        accuracy_per_epoch = []
        i = 0
        total = 0
        batch_accuracy_ = 0
        total_ = 0
        accuracy = 0
        running_corrects = 0

        for i, train_data in tqdm(enumerate(train_generator)):

            x_train, y_train = train_data['image'].to(device, non_blocking=True), train_data['label'].to(device, non_blocking=True)
            y_train = y_train.unsqueeze(1).float()

            optimizer.zero_grad()

            output = model(x_train)

            train_loss = criterion(input=output.to(torch.float32), target=y_train.to(torch.float32))

            train_loss.backward()
            optimizer.step()
            num_data = y_train.size()[0]
            preds = torch.argmax(output, dim=1)
            correct_pred = torch.sum(preds == y_train)
            batch_accuracy = correct_pred * (100/num_data)

            _, predicted = torch.max(output, 1)
            total += y_train.size(0)
            batch_accuracy_ += (predicted == y_train).sum().item()
            pred = torch.softmax(output, dim=1)
            accuracy += (pred == y_train).sum().item()
            total_ += y_train.numel()

            loss_per_epoch.append(train_loss.item())
            accuracy_per_epoch.append(batch_accuracy.item())

            running_corrects += torch.sum(preds == y_train.squeeze())

        curr_train_loss = sum(loss_per_epoch) / (i + 1)

        epoch_acc = running_corrects / (len(train_generator) * args["batch_size"])

        train_loss_list.append(curr_train_loss)
        train_acc_list.append(epoch_acc)

        print(f"Epoch {epoch + 1}/{args['epoch']}")
        print(f"Train_loss :  {curr_train_loss:.5f} Train accuracy:  {epoch_acc:.5f}")

        model.eval()
        loss_per_epoch = []
        accuracy_per_epoch = []
        i = 0
        total = 0
        batch_accuracy_ = 0
        total_ = 0
        accuracy = 0
        running_corrects = 0

        with torch.no_grad():
            for i, test_data in tqdm(enumerate(test_generator)):
                x_test, y_test = test_data['image'].to(device, non_blocking=True), test_data['label'].to(device, non_blocking=True)
                y_test = y_test.unsqueeze(1).float()

                output = model(x_test)

                test_loss = criterion(input=output.to(torch.float32), target=y_test.to(torch.float32))

                num_data = y_test.size()[0]
                preds = torch.argmax(output, dim=1)

                correct_pred = torch.sum(preds == y_test)
                batch_accuracy = correct_pred * (100/num_data)
                batch_accuracy = batch_accuracy.item()

                _, predicted = torch.max(output, 1)
                total += y_test.size(0)
                batch_accuracy_ += (predicted == y_test).sum().item()

                loss_per_epoch.append(test_loss.item())
                accuracy_per_epoch.append(batch_accuracy)

                running_corrects += torch.sum(preds == y_test.squeeze())

                pred = torch.softmax(output, dim=1)

                accuracy += (pred == y_test).sum().item()
                total_ += y_test.numel()

                y_pred.append(output.detach().cpu().numpy())
                y_true.append(y_test.detach().cpu().numpy())

            curr_test_accuracy = sum(accuracy_per_epoch) / (i + 1)
            curr_test_loss = sum(loss_per_epoch) / (i + 1)

            epoch_acc = running_corrects / (len(test_generator) * args["batch_size"])

            test_loss_list.append(curr_test_loss)
            test_acc_list.append(epoch_acc)

            valid_losses.append(curr_test_loss)

        print(f"Validation loss: {curr_test_loss:.5f} Validation accuracy: {epoch_acc:.5f}")

        if epoch % 5 == 0:
            lr_decay.step()
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}")

        if best_accuracy < curr_test_accuracy:
            torch.save(model.state_dict(), f"{model_save_folder}UBI-FIGHTS_640x360_videocapture_densenet_cbam_lstm.pth")
            best_accuracy = curr_test_accuracy
            print('Current model has best valid accuracy')

        valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(type(y_pred), type(y_true))
    y_pred = np.array(y_pred, dtype=np.int64)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)

    print(type(y_pred), type(y_true))
    print(y_pred.shape, y_true.shape)

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    print(y_pred.shape, y_true.shape)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro') # , multi_class='ovr'
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro') # , multi_class='ovr'
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"]+f"densenet_cbam_lstm_auc.png")
    plt.show()

    print('AUC: ', roc_auc)

    train_acc_list = torch.Tensor(train_acc_list).detach().cpu().numpy().tolist()
    test_acc_list = torch.Tensor(test_acc_list).detach().cpu().numpy().tolist()
    train_loss_list = torch.Tensor(train_loss_list).detach().cpu().numpy().tolist()
    test_loss_list = torch.Tensor(test_loss_list).detach().cpu().numpy().tolist()

    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.title('Train and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"]+f"/densenet_cbam_lstm_acc.png")
    plt.show()

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('Train and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"]+f"/densenet_cbam_lstm_loss.png")
    plt.show()


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    train(args=args, device=device)

    ### 지금 이게 아나콘다 파워쉘 프롬프트 마지막꺼에서 돌아가고 있고 ubi-fights 640,360 데이터 이미지로
    # fight가 0 normal이 1