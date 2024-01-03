import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
# from torchvision import transforms

from sklearn.metrics import f1_score, accuracy_score, auc, roc_auc_score, roc_curve

import cv2, os, glob
from tqdm import tqdm
import matplotlib.pyplot as plt

args = {"data_folder": "D:\\abnormal_detection_dataset\\UBI_FIGHTS\\videos\\videos\\",
        "graphs_folder": "./graph/UBI-FIGHTS_video/", "epoch": 50, "batch_size": 8, "num_class": 1, "learning_rate": 0.001,  # 원래 1e-4
        "decay_rate": 0.998, "num_workers": 4, "img_size": (360, 240), "img_depth": 3, "FPS": 30} # decay_rate 원래 0.98 "img_size": (320, 240)
model_save_folder = './trained_model/UBI-FIGHTS_video/'


device = torch.device('cuda:0')
## 2d
############################################################### 2d cbam ###################################################
# class Channel_Attention(nn.Module):
#
#     def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
#         #  channel_in: 입력 피쳐 맵의 채널 수를 나타내는 매개변수로 전달
#         #  reduction_ratio: 채널 어텐션에서 사용되는 감소 비율(reduction ratio)을 나타내는 매개변수로, 기본값은 16
#         super(Channel_Attention, self).__init__()
#         self.pool_types = pool_types
#
#         self.shared_mlp = nn.Sequential(  # self.shared_mlp은 채널 어텐션에서 사용되는 공유 다층 퍼셉트론(MLP)을 나타내는 nn.Sequential 모듈
#             nn.Flatten(),
#             nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),  # 입력 피쳐 맵의 채널 수를 감소 비율에 따라 줄이는 fully-connected 레이어
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)   # 감소된 채널 수를 원래의 채널 수로 확장하는 fully-connected 레이어
#         )
#
#
#     def forward(self, x):
#
#         channel_attentions = []
#
#         for pool_types in self.pool_types:
#             if pool_types == 'avg':
#                 # pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 2d
#                 pool_init = nn.AvgPool3d(kernel_size=(1, x.size(2), x.size(3)), stride=(1, x.size(2), x.size(3)))  # 3d
#                 avg_pool = pool_init(x)
#                 channel_attentions.append(self.shared_mlp(avg_pool))  # x에 대해 avgpool2d 연산 후 avg_pool 변수에 저장하고 그걸 __init__의 self.shared_mlp 통과시킨 뒤 channel_attentions 리스트에 추가
#             elif pool_types == 'max':
#                 # pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 위와 같고 maxpooling2d 연산을 한다는 것만 다름  2d
#                 pool_init = nn.AvgPool3d(kernel_size=(1, x.size(2), x.size(3)), stride=(1, x.size(2), x.size(3)))  # 3d
#                 max_pool = pool_init(x)
#                 channel_attentions.append(self.shared_mlp(max_pool))
#
#         pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)  # 각 풀링에 대한 channel attention 결과를 합쳐서(가장 앞 차원 의미하는 dim=0 기준으로 결과 합침)pooling_sums 변수에 저장
#         # 그 후 .sum(dim=0)을 통해 첫 번째 차원을 따라 합산을 수행하여 각 위치에 대해 channel_attentions 리스트의 원소들을 합친 결과를 얻는다.
#         # 이렇게 얻어진 pooling_sums 텐서에는 avg와 max 풀링의 결과에 대한 channel attention 값들이 합산되어 저장되어 있다., pooling_sums 텐서의 크기는 (batch_size, channel)
#         # scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)  # nn.Sigmoid()를 통해 합쳐진 채널 어텐션 결과에 시그모이드 함수를 적용하여 0~1 값으로 변환 > channel attention의 스케일링을 나타냄.  2d
#         # unsqueeze(2).unsqueeze(3)를 통해 pooling_sums 텐서의 크기는 (batch_size, channel, 1, 1)이 됨. 최종 결과: 입력 데이터 x와 동일한 크기로 확장
#         # .expand_as(x) 통해 pooling_sums 텐서의 크기를 x 텐서와 동일하게 확장함. x 텐서는 입력으로 주어진 텐서로, pooling_sums 텐서와 같은 크기를 가지게 된다.
#         # 이렇게 얻어진 scaled 텐서에는 pooling_sums 텐서의 값을 x 텐서와 동일한 크기로 확장한 결과가 저장되어 있다.
#         # scaled 텐서의 크기는 x 텐서((batch_size, channel_in, height, width))와 동일한 크기를 가지게 된다.
#         ############ scaled 텐서의 크기는 (batch_size, channel, height, width)
#         # unsqueeze(2)와 unsqueeze(3)를 통해 pooling_sums 텐서에 2개의 차원을 추가하고, expand_as(x)를 통해 x 텐서와 동일한 크기로 확장
#         scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).unsqueeze(4)
#         scaled = scaled.expand_as(x)
#
#         return x * scaled  # return the element-wise multiplication between the input and the result.
#         # x와 scaled를 element-wise 곱한 결과를 반환하는 부분이고, x는 원본 입력 텐서이고, scaled는 채널 어텐션을 적용한 결과이다
#         # 두 텐서의 크기가 동일하므로 element-wise 곱셈이 가능하다. 이 연산을 통해 입력 텐서 x에 채널 어텐션을 적용한 결과를 얻을 수 있다.
#
# class ChannelPool(nn.Module):
#     # 피쳐맵의 모든 채널들을 두 개의 개별 채널로 병합. 첫 번째 채널은 모든 채널에서 !최댓값!을 가져와 생성하고 두 번째 채널은 모든 채널에서 !평균!을 가져와 생성한다.
#     def forward(self, x):  # 입력 피쳐 맵의 모든 채널을 최대값과 평균값을 사용하여 두 개의 채널로 병합하는 기능
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#     # torch.max(x, 1)[0].unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 최대값을 계산하고, 최대값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
#     # torch.mean(x, 1).unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 평균값을 계산하고, 평균값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
#     # torch.cat, dim=1: 위에서 얻은 두 개의 텐서를 채널 차원을 기준으로 병합한다. 이를 통해 최대값과 평균값을 각각 따로 가진 두 개의 채널을 병합한 결과를 얻을 수 있다.
#
# # class Spatial_Attention(nn.Module):  # 2d
# #
# #     def __init__(self, kernel_size=7): # kernel_size를 인자로 받음
# #
# #         # Spatial_Attention 아키텍처
# #         super(Spatial_Attention, self).__init__()
# #
# #         self.compress = ChannelPool() # 위의 ChannelPool 객체 생성
# #         self.spatial_attention = nn.Sequential(  # 객체 생성
# #             nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
# #             nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
# #         )
# #
# #
# #     def forward(self, x):
# #         x_compress = self.compress(x)  # ChannelPool 객체인 self.compress(x) 호출해 입력 텐서에 대해 연산
# #         x_output = self.spatial_attention(x_compress)  # 위의 연산 결과를 spatial_attention을 호출해 Spatial Attention 적용한 결과 x_output에 저장
# #         scaled = nn.Sigmoid()(x_output)  # sigmoid 통해 값을 0~1로 만듦
# #         return x * scaled  # x와 활성화 값인 scaled를 곱하여 element-wise 곱셈을 수행한 후, 그 결과를 반환: 입력 텐서 x를 활성화 값 scaled로 가중치를 적용하여 출력을 생성하는 동작임.
# class Spatial_Attention(nn.Module): ## 3d
#
#     def __init__(self, kernel_size=7):
#         super(Spatial_Attention, self).__init__()
#
#         self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(1, kernel_size, kernel_size), stride=(1, 1, 1),
#                               padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
#         self.bn = nn.BatchNorm3d(1)
#         self.act_fn = nn.ReLU(inplace=True)
#
#
#     def forward(self, x):
#         # 입력 텐서 크기를 (batch_size, channels, depth, height, width)로 변경
#         x = x.unsqueeze(2)
#         x_compress = torch.cat((torch.max(x, 2)[0].unsqueeze(2), torch.mean(x, 2).unsqueeze(2)), dim=2)
#
#         # Conv2d 대신 Conv3d를 사용하여 합성곱 연산 수행
#         x_out = self.conv(x_compress)
#         x_out = self.bn(x_out)
#         x_out = self.act_fn(x_out)
#
#         # 5차원 텐서를 다시 4차원으로 변경
#         x_out = x_out.squeeze(2)
#
#         return x_out  # 3d
#
#
# class CBAM(nn.Module):
#     # CBAM 아키텍처
#     # def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
#     def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
#         super(CBAM, self).__init__()  # 상속
#         # self.spatial = spatial # Spatial channel의 활성화 여부를 나타냄. boolean 값
#
#         self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types) # 채널 어텐션 모듈 객체를 생성
#
#         # if self.spatial: # spatial=True 이면
#         self.spatial_attention = Spatial_Attention(kernel_size=7) # Spatial_Attention 생성
#
#
#     def forward(self, x):
#         x_out = self.channel_attention(x) # 입력 데이터 x에 대해 채널 어텐션 적용 후 x_out에 저장
#         # if self.spatial: # spatial=True 이면
#         x_out = self.spatial_attention(x_out) # 위의 x_out에 Spatial_Attention 적용
#
#         return x_out
############################################################### 2d cbam ###################################################

class CBAM(nn.Module):
    def __init__(self, in_channels=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        # self.spatial_attention = SpatialAttention(in_channels)
        self.spatial_attention = SpatialAttention(channel=64)

    def forward(self, x):
        x_out = self.channel_attention(x)
        x_out = self.spatial_attention(x_out)
        return x_out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = self.fc1(avg_pool)
        max_pool = self.fc1(max_pool)
        avg_pool = self.relu(avg_pool)
        max_pool = self.relu(max_pool)
        avg_pool = self.fc2(avg_pool)
        max_pool = self.fc2(max_pool)
        x_out = self.sigmoid(avg_pool + max_pool)
        return x_out

class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction_ratio=8):
        super(SpatialAttention, self).__init__()
        self.conv1 = Conv3d_BN(channel, channel // reduction_ratio, kernel_size=1, bias=False)
        self.conv2 = Conv3d_BN(channel // reduction_ratio, channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_out = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)   # 수정해야 될수도 있음
        x_out = self.conv1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.sigmoid(x_out)
        return x * x_out

class Conv3d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(Conv3d_BN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Unit3D(nn.Module):
    """
    3D Convolutional layer with batch normalization and activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Unit3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class I3D(nn.Module):
    """
    Inflated 3D ConvNet (I3D) model architecture
    """
    def __init__(self, num_classes=args["num_class"], dropout_rate=0.3):
        super(I3D, self).__init__()
        self.dropout_rate = dropout_rate

        # stem
        self.stem = nn.Sequential(
            Unit3D(in_channels=3, out_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Unit3D(in_channels=64, out_channels=64, kernel_size=(1, 1, 1), stride=1, padding=0),
            Unit3D(in_channels=64, out_channels=192, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # mixed 3b
        self.mixed_3b = nn.Sequential(
            Unit3D(in_channels=192, out_channels=64, kernel_size=(1, 1, 1), stride=1, padding=0),
            Unit3D(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), stride=1, padding=1),
            Unit3D(in_channels=96, out_channels=96, kernel_size=(3, 3, 3), stride=1, padding=1)
        )

        # mixed 3c
        self.mixed_3c = nn.Sequential(
            Unit3D(in_channels=96, out_channels=64, kernel_size=(1, 1, 1), stride=1, padding=0),
            Unit3D(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), stride=1, padding=1),
            Unit3D(in_channels=96, out_channels=96, kernel_size=(3, 3, 3), stride=1, padding=1)
        )

        # mixed 4a
        self.mixed_4a = nn.Sequential(
            Unit3D(in_channels=96, out_channels=64, kernel_size=(1, 1, 1), stride=1, padding=0),
            Unit3D(in_channels=64, out_channels=96, kernel_size=(1, 3, 3), stride=1, padding=1),
            Unit3D(in_channels=96, out_channels=96, kernel_size=(3, 1, 1), stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # mixed 4b
        self.mixed_4b = self._make_mixed_block(in_channels=96, out_channels=[96, 208, 16, 208])

        # mixed 4c
        self.mixed_4c = self._make_mixed_block(in_channels=208, out_channels=[96, 208, 16, 256])

        # mixed 4d
        self.mixed_4d = self._make_mixed_block(in_channels=256, out_channels=[96, 208, 16, 256])

        # mixed 4e
        self.mixed_4e = self._make_mixed_block(in_channels=256, out_channels=[96, 208, 16, 256])

        # mixed 4f
        self.mixed_4f = self._make_mixed_block(in_channels=256, out_channels=[96, 208, 16, 256])

        # mixed 5a
        self.mixed_5a = self._make_mixed_block(in_channels=256, out_channels=[128, 256, 32, 512])

        # mixed 5b
        self.mixed_5b = self._make_mixed_block(in_channels=512, out_channels=[128, 256, 32, 64])

        self.cbam = CBAM(in_channels=64)

        # global average pooling and dropout
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # output
        self.fc = nn.Sequential(nn.Linear(in_features=64, out_features=args["num_class"]), nn.Sigmoid())

    def forward(self, x):
        x = self.stem(x)            # torch.Size([2, 192, 75, 30, 40])
        # print(x.shape)
        x = self.mixed_3b(x)        # torch.Size([2, 96, 75, 30, 40])
        # print(x.shape)
        x = self.mixed_3c(x)        # torch.Size([2, 96, 75, 30, 40])
        # print(x.shape)
        x = self.mixed_4a(x)        # torch.Size([2, 96, 77, 16, 21])
        # print(x.shape)
        x = self.mixed_4b(x)        # torch.Size([2, 208, 39, 8, 11])
        # print(x.shape)
        x = self.mixed_4c(x)        # torch.Size([2, 256, 20, 4, 6])
        # print(x.shape)
        x = self.mixed_4d(x)        # torch.Size([2, 256, 10, 2, 3])
        # print(x.shape)
        x = self.mixed_4e(x)        # torch.Size([2, 256, 5, 1, 2])
        # print(x.shape)
        x = self.mixed_4f(x)        # torch.Size([2, 256, 3, 1, 1])
        # print(x.shape)
        x = self.mixed_5a(x)        # torch.Size([2, 512, 2, 1, 1])
        # print(x.shape)
        x = self.mixed_5b(x)        # torch.Size([2, 64, 1, 1, 1])
        # print(x.shape)
        x = self.cbam(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.avgpool(x)         # torch.Size([2, 64, 1, 1, 1])
        # print(x.shape)
        x = self.dropout(x)         # torch.Size([2, 64, 1, 1, 1])
        # print(x.shape)
        # x = x.mean(dim=1)
        # print(x.shape)
        x = x.view(x.size(0), -1)   # torch.Size([2, 96, 75, 30, 40])
        # print(x.shape)

        # print(x.shape)
        x = self.fc(x)              # torch.Size([2, 64])
        x = self.dropout(x)
        # print(x.shape)
        return x

    def _make_mixed_block(self, in_channels, out_channels):
        """
        Mixed block
        """
        return nn.Sequential(
            Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel_size=(1, 1, 1), stride=1, padding=0),
            Unit3D(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3, 3, 3), stride=1, padding=1),
            Unit3D(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(1, 1, 1), stride=1, padding=0),
            Unit3D(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=(1, 1, 1), stride=1, padding=0),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        )

class CustomDataset(Dataset):

    def __init__(self, dataset_folder_path=args["data_folder"], image_size=args["img_size"], image_depth=3, train=True, transform=None):
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform  # HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
        # self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomHorizontalFlip(p=0.7),
        #                                      transforms.RandomVerticalFlip(p=0.7), transforms.RandomHorizontalFlip(), transforms.RandomRotation(180, expand=False)])
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()

    def __len__(self):
        # print(self.image_path_label)
        return len(self.image_path_label)

    def __getitem__(self, idx):
        frames = []
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, label = self.image_path_label[idx]
        cap = cv2.VideoCapture(image)
        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print('fps: ', fps)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(fps, width, height)

        # for frame in range(args["FPS"]):
        for frame in range(int(fps)):
            frame, image = cap.read()
            # print('_-_-_-__-_-_-__-_-_-__-_-_-__-_-_-__-_-_-__-_-_-_')
            # print('fps: ', fps)
            cnt = 0
            if self.image_depth == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, args["img_size"])
                # print('color image size: ', image.shape)  # (240, 320, 3)
                # print(width, height)
                # print(self.read_folder(self.train))
                image = image / 255
                frames.append(image)
                # cv2.imshow('img', image)
                # print(image.shape)
                # cv2.waitKey()
                # print(np.array(frames).shape)
                # frames.append(image)
                # print('dnl', image.shape)
                # print(label)
                # cv2.imshow('3', image)
                # cv2.waitKey()
            else:
                image = cv2.resize(image, args["img_size"])
                image = image / 255
                frames.append(image)
            # print(np.array(frames).shape)
            # print(torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2).shape)  # torch.Size([3, 32, 240, 320])
                # cv2.imshow('img', image)
                # cv2.waitKey()
                # print(np.array(frames).shape)
                # print(image.shape)

                # print(image.shape)
                # print(label)
                # cv2.imshow('0', image)
                # cv2.waitKey()
        # if self.image_depth == 1:
        #     image = cv2.imread(image, 0)
        # else:
        #     image = cv2.imread(image)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if self.transform:
        #     image = self.transform(image)
        #         frames.append(image)
        # frames = np.array(frames)
        # print(frames)
        # print('=============================')
        # a = torch.tensor(frames)
        # print(a.shape)

        # return image, label
        # return {'image': image, 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
        # return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2), {'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
        # a = torch.tensor(frames)
        # print(a)
        # print(frames)
        # a = np.array(frames)
        # print(a.shape)pr
        # print('frames', np.array(frames).shape, label)
        # return {'image': torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장

        # return {'image': torch.FloatTensor(image).permute(2, 0, 1), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장

        return {'image': torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장
        # return {'image': torch.FloatTensor(np.array(frames)), 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장

    def get_classnames(self):
        # print('---------------------------')
        # print(os.listdir(f"{self.dataset_folder_path}/train/"))   # ['Fight', 'NonFight'],  UBI-FIGHTS: ['fight', 'normal']
        return os.listdir(f"{self.dataset_folder_path}train/")  # ['Fight', 'NonFight']

    def read_folder(self):
        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path}train/"
            # print('train folder_path: ', folder_path)
        else:
            folder_path = f"{self.dataset_folder_path}val/"  # valid도 가능
            # print('val folder_path: ', folder_path)

        for x in glob.glob(folder_path + "**", recursive=True):  # 해당 폴더의 하위 폴더까지 탐색
            if not x.endswith('mp4'): # rwf2000: avi, ubi-fights: mp4
                continue
            # print("x.split('\\')", x.split('\\'))
            class_idx = self.classes.index(x.split('\\')[-2])  # 클래스 이름(Violence, Nonviolence)의 인덱스 저장 # ubi, ucfcrime
            # print('class_idx: ', class_idx, ' ', 'classes.index(x.split("\\")[-2]: ', x.split('\\')[-2])
            # print(x, x.split('\\')[-2], class_idx) # D:\abnormal_detection_dataset\RWF_2000_Dataset\train\Fight\fi98.avi Fight 0
            ############################## Fight가 0, NonFight가 1 ############################## : RWF2000
            ############################## fight가 0, normal이 1 ############################## : UBI-FIGHTS
            image_path_label.append(
                (x, int(class_idx)))  # ('D:\\abnormal_detection_dataset\\RWF_2000_Dataset\\train\\Fight\\fi243.avi', 0)
            # for i in image_path_label:
            #     if 1 in image_path_label:
            #         print(i)
            #     else:
            #         print('fu')
            # print('image_path_label: ', image_path_label) # 'D:\\abnormal_detection_dataset\\RWF_2000_Dataset\\train\\Fight\\fi243.avi', 0) # Fight:0, NonFight: 1 UBI-FIGHTS도 동일
        return image_path_label  # (이미지 경로, 클래스 인덱스)

##### 콜백함수
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path=f"{model_save_folder}RWF2000_model_early.pth"):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
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
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  val loss is decreased, Saving model ...')
            print('\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

##### 가중치 초기화 함수
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def train(args, device:"cuda:0"):
    model = I3D(num_classes=args["num_class"], dropout_rate=0.3)
    initialize_weights(model)

    model.to(device)
    # summary(model, (3, 360, 240))

    # 옵티마이저 정의
    # optimizer = optim.SGD(model.parameters(), lr=args["learning_rate"], momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer = Adam(model.parameters(), args["learning_rate"])
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args["learning_rate"])
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

    # lr_decay = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args["decay_rate"])
    lr_decay = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    # 손실 함수 정의
    criterion = nn.BCEWithLogitsLoss().to(device)

    ##### train, test dataset 정의
    train_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                                  image_depth=args["img_depth"],
                                  train=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                                 transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(180, expand=False)]))  # transforms.ToTensor(): HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
                                  # transforms=transforms.ToTensor())  # transforms.ToTensor(): HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
    test_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                                 image_depth=args["img_depth"],
                                 train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # 학습 데이터 로더 정의
    train_generator = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,
                                 num_workers=args["num_workers"], pin_memory=True)  # , sampler=sampler
    test_generator = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True,
                                num_workers=args["num_workers"],
                                pin_memory=True)

    early_stopping = EarlyStopping(verbose=True)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    best_accuracy = 0
    valid_losses = []
    y_true = []
    y_pred = []

    # 학습
    for epoch in range(args["epoch"]):
        model.train()
        loss_per_epoch = []
        accuracy_per_epoch = []

        for i, train_data in tqdm(enumerate(train_generator)):

            inputs, labels = train_data['image'], train_data['label'] # torch.Size([16, 3, 150, 240, 320])
            # print(inputs.shape)
            # print(labels.shape)
            inputs = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4) # torch.Size([150, 3, 16, 240, 320])
            # print(labels.shape)

            # print(inputs.shape)
            # inputs, labels = train_data['image'].to(device, non_blocking=True), train_data['label'].to(device, non_blocking=True)
            inputs, labels = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4).to(device, non_blocking=True), labels.to(device, non_blocking=True)
            labels = labels.unsqueeze(1)


            # print(labels.shape)
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            # print(inputs.shape, labels.shape, outputs.shape)
            train_loss = criterion(input=outputs.to(torch.float32), target=labels.to(torch.float32))

            # 역전파
            train_loss.backward()
            optimizer.step()

            num_data = labels.size()[0]
            preds = torch.argmax(outputs, dim=1)
            correct_pred = torch.sum(preds == labels)
            # batch_accuracy = correct_pred * (100/num_data)  # accuracy 이상하게 나옴
            batch_accuracy = correct_pred * (100/len(labels))

            loss_per_epoch.append(train_loss.item())
            accuracy_per_epoch.append(batch_accuracy.item())

            # threshold = 0.5                                     # Train accuracy: , 1.00000
            # predicted = (outputs > threshold).float()           # Train accuracy: , 1.00000
            # accuracy = (predicted == labels).float().mean()     # Train accuracy: , 1.00000

            threshold = 0.5
            predicted = (outputs > threshold).float()
            correct = (predicted == labels).float().sum()
            accuracy = correct / len(labels)

        curr_train_accuracy = sum(accuracy_per_epoch) / (i + 1)
        curr_train_loss = sum(loss_per_epoch) / (i + 1)

        train_loss_list.append(curr_train_loss)
        train_acc_list.append(accuracy)

        print(f"Epoch {epoch + 1}/{args['epoch']}")
        # print('=' * 100)
        # print(f"Train Loss : {curr_train_loss:.5f}, Train accuracy : {(batch_accuracy_ / total):.5f}")
        # print('=' * 100)
        # print(f"Training Loss : {curr_train_loss}, Training accuracy : {train_acc:.5f}")
        print(f"Train_loss :  {curr_train_loss:.5f}, 'Train accuracy: , {accuracy:.5f}")

        model.eval()
        loss_per_epoch = []
        accuracy_per_epoch = []

        with torch.no_grad():
            for i, test_data in tqdm(enumerate(test_generator)):
                inputs, labels = test_data['image'], test_data['label']
                inputs = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4)

                # inputs, labels = test_data['image'].to(device, non_blocking=True), test_data['label'].to(device, non_blocking=True)
                # inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                inputs, labels = torch.FloatTensor(np.array(inputs)).permute(2, 1, 0, 3, 4).to(device, non_blocking=True), labels.to( device, non_blocking=True)

                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                valid_loss = criterion(input=outputs.to(torch.float32), target=labels.to(torch.float32))

                num_data = labels.size()[0]
                preds = torch.argmax(outputs, dim=1)
                correct_pred = torch.sum(preds == labels)
                # batch_accuracy = correct_pred * (100 / num_data)  # accuracy 이상하게 나옴
                batch_accuracy = correct_pred * (100 / len(labels))

                loss_per_epoch.append(valid_loss.item())
                accuracy_per_epoch.append(batch_accuracy)

                # threshold = 0.5                                   # Valid accuracy: , 1.00000
                # predicted = (outputs > threshold).float()         # Valid accuracy: , 1.00000
                # accuracy = (predicted == labels).float().mean()   # Valid accuracy: , 1.00000

                threshold = 0.5
                predicted = (outputs > threshold).float()
                correct = (predicted == labels).float().sum()
                accuracy = correct / len(labels)

                # 이진 분류 AUC 구하기
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

            curr_test_accuracy = sum(accuracy_per_epoch) / (i + 1)
            curr_test_loss = sum(loss_per_epoch) / (i + 1)

            test_loss_list.append(curr_test_loss)
            test_acc_list.append(accuracy)

            valid_losses.append(curr_test_loss)

        print(f"Valid loss :  {curr_test_loss:.5f}, 'Valid accuracy: , {accuracy:.5f}")

        if epoch % 5 == 0:
            lr_decay.step()
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}")

        if best_accuracy < curr_test_accuracy:
            torch.save(model.state_dict(), f"{model_save_folder}UBI-FIGHTS_640x360_videocapture_model.pth")
            best_accuracy = curr_test_accuracy
            print('Current model has best valid accuracy')

        # print('\n--------------------------------------------------------------------------------\n')
        valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(type(train_acc_list))
    # train_acc_list = np.array(train_acc_list).detach().cpu().numpy().tolist()
    # test_acc_list = np.array(test_acc_list).detach().cpu().numpy().tolist()
    # train_loss_list = np.array(train_loss_list).detach().cpu().numpy().tolist()
    # test_loss_list = np.array(test_loss_list).detach().cpu().numpy().tolist()
    print('train_acc_list: ', '\n', train_acc_list)
    print('test_acc_list: ', '\n', test_acc_list)
    print('train_loss_list: ', '\n', train_loss_list)
    print('test_loss_list: ', '\n', test_loss_list)
    train_acc_list = torch.Tensor(train_acc_list).detach().cpu().numpy().tolist()
    test_acc_list = torch.Tensor(test_acc_list).detach().cpu().numpy().tolist()
    train_loss_list = torch.Tensor(train_loss_list).detach().cpu().numpy().tolist()
    test_loss_list = torch.Tensor(test_loss_list).detach().cpu().numpy().tolist()

    print(type(train_acc_list))
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.title('Train and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"]+f"UBI-FIGHTS_video_acc_graph.png")
    plt.show()

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('Train and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"]+f"UBI-FIGHTS_video_loss_graph.png")
    plt.show()

    # gpt auc 시작
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro')
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
    plt.title('Receiver Operating Characteristic!')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + f"UBI-FIGHTS_videoAUC_graph.png")
    plt.show()

    print('AUC: ', roc_auc)
    # gpt auc 끝


    # 내가 쓰던 auc
    print(type(y_pred), type(y_true))  # <class 'list'> <class 'list'>
    y_pred = np.array(y_pred, dtype=np.int64)  # .sum(axis=0)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)  # .max(axis=1)
    print(y_pred.shape, y_true.shape)
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    print(y_pred.shape, y_true.shape)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro')  # , multi_class='ovr'
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')  # , multi_class='ovr'
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + f"UBI-FIGHTS_videoAUC_graph!.png")
    plt.show()

    print('AUC: ', roc_auc)
    # 내가 쓰던 auc 끝


if __name__ == '__main__':
    train(args=args, device=device)