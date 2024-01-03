# 임포트 import
import os
from tqdm import tqdm
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, auc, roc_auc_score, roc_curve



args = {"data_folder": "D:\\abnormal_detection_dataset\\UBI_FIGHTS\\for_train\\",
        "graphs_folder": "./graph_rwf2000", "epoch": 1, "batch_size": 64, "num_class": 2, "learning_rate": 1e-4,
        "decay_rate": 0.98, "num_workers": 4, "img_size": (640, 360), "img_depth": 3}

# gpu 설정
device = torch.device('cuda:0')

# 그래프 plot 폴더 경로 설정
if not os.path.exists(args["graphs_folder"]) : os.mkdir(args["graphs_folder"])
# model_save_folder = 'resnet_cbam/' if args.use_cbam else 'resnet/'
model_save_folder = './trained_model/RWF2000'

if not os.path.exists(model_save_folder) : os.mkdir(model_save_folder)


##### CBAM 정의  Channel_Attnetion > ChannelPool > Spatial_Attention > CBAM 순으로 정의
class Channel_Attention(nn.Module):

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        #  channel_in: 입력 피쳐 맵의 채널 수를 나타내는 매개변수로 전달
        #  reduction_ratio: 채널 어텐션에서 사용되는 감소 비율(reduction ratio)을 나타내는 매개변수로, 기본값은 16
        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types

        self.shared_mlp = nn.Sequential(  # self.shared_mlp은 채널 어텐션에서 사용되는 공유 다층 퍼셉트론(MLP)을 나타내는 nn.Sequential 모듈
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),  # 입력 피쳐 맵의 채널 수를 감소 비율에 따라 줄이는 fully-connected 레이어
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)   # 감소된 채널 수를 원래의 채널 수로 확장하는 fully-connected 레이어
        )


    def forward(self, x):

        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))  # x에 대해 avgpool2d 연산 후 avg_pool 변수에 저장하고 그걸 __init__의 self.shared_mlp 통과시킨 뒤 channel_attentions 리스트에 추가
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 위와 같고 maxpooling2d 연산을 한다는 것만 다름
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)  # 각 풀링에 대한 channel attention 결과를 합쳐서(가장 앞 차원 의미하는 dim=0 기준으로 결과 합침)pooling_sums 변수에 저장
        # 그 후 .sum(dim=0)을 통해 첫 번째 차원을 따라 합산을 수행하여 각 위치에 대해 channel_attentions 리스트의 원소들을 합친 결과를 얻는다.
        # 이렇게 얻어진 pooling_sums 텐서에는 avg와 max 풀링의 결과에 대한 channel attention 값들이 합산되어 저장되어 있다., pooling_sums 텐서의 크기는 (batch_size, channel)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)  # nn.Sigmoid()를 통해 합쳐진 채널 어텐션 결과에 시그모이드 함수를 적용하여 0~1 값으로 변환 > channel attention의 스케일링을 나타냄.
        # unsqueeze(2).unsqueeze(3)를 통해 pooling_sums 텐서의 크기는 (batch_size, channel, 1, 1)이 됨. 최종 결과: 입력 데이터 x와 동일한 크기로 확장
        # .expand_as(x) 통해 pooling_sums 텐서의 크기를 x 텐서와 동일하게 확장함. x 텐서는 입력으로 주어진 텐서로, pooling_sums 텐서와 같은 크기를 가지게 된다.
        # 이렇게 얻어진 scaled 텐서에는 pooling_sums 텐서의 값을 x 텐서와 동일한 크기로 확장한 결과가 저장되어 있다.
        # scaled 텐서의 크기는 x 텐서((batch_size, channel_in, height, width))와 동일한 크기를 가지게 된다.
        ############ scaled 텐서의 크기는 (batch_size, channel, height, width)
        # unsqueeze(2)와 unsqueeze(3)를 통해 pooling_sums 텐서에 2개의 차원을 추가하고, expand_as(x)를 통해 x 텐서와 동일한 크기로 확장

        return x * scaled  # return the element-wise multiplication between the input and the result.
        # x와 scaled를 element-wise 곱한 결과를 반환하는 부분이고, x는 원본 입력 텐서이고, scaled는 채널 어텐션을 적용한 결과이다
        # 두 텐서의 크기가 동일하므로 element-wise 곱셈이 가능하다. 이 연산을 통해 입력 텐서 x에 채널 어텐션을 적용한 결과를 얻을 수 있다.


class ChannelPool(nn.Module):
    # 피쳐맵의 모든 채널들을 두 개의 개별 채널로 병합. 첫 번째 채널은 모든 채널에서 !최댓값!을 가져와 생성하고 두 번째 채널은 모든 채널에서 !평균!을 가져와 생성한다.
    def forward(self, x):  # 입력 피쳐 맵의 모든 채널을 최대값과 평균값을 사용하여 두 개의 채널로 병합하는 기능
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
    # torch.max(x, 1)[0].unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 최대값을 계산하고, 최대값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
    # torch.mean(x, 1).unsqueeze(1): 입력 텐서 x의 모든 채널에 대해서 평균값을 계산하고, 평균값들을 채널 차원에 대해 제거한 결과를 반환한 것을 2d 텐서로 만듦
    # torch.cat, dim=1: 위에서 얻은 두 개의 텐서를 채널 차원을 기준으로 병합한다. 이를 통해 최대값과 평균값을 각각 따로 가진 두 개의 채널을 병합한 결과를 얻을 수 있다.


class Spatial_Attention(nn.Module):

    def __init__(self, kernel_size=7): # kernel_size를 인자로 받음

        # Spatial_Attention 아키텍처
        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool() # 위의 ChannelPool 객체 생성
        self.spatial_attention = nn.Sequential(  # 객체 생성
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )


    def forward(self, x):
        x_compress = self.compress(x)  # ChannelPool 객체인 self.compress(x) 호출해 입력 텐서에 대해 연산
        x_output = self.spatial_attention(x_compress)  # 위의 연산 결과를 spatial_attention을 호출해 Spatial Attention 적용한 결과 x_output에 저장
        scaled = nn.Sigmoid()(x_output)  # sigmoid 통해 값을 0~1로 만듦
        return x * scaled  # x와 활성화 값인 scaled를 곱하여 element-wise 곱셈을 수행한 후, 그 결과를 반환: 입력 텐서 x를 활성화 값 scaled로 가중치를 적용하여 출력을 생성하는 동작임.



class CBAM(nn.Module):
    # CBAM 아키텍처
    # def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']): # CBAM 클래스의 생성자 함수, 인자로 channel_in(입력 채널 수), reduction_ratio(채널 축소 비율), 풀링 타입, Spatial channel의 활성화 여부
        super(CBAM, self).__init__()  # 상속
        # self.spatial = spatial # Spatial channel의 활성화 여부를 나타냄. boolean 값

        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types) # 채널 어텐션 모듈 객체를 생성

        # if self.spatial: # spatial=True 이면
        self.spatial_attention = Spatial_Attention(kernel_size=7) # Spatial_Attention 생성


    def forward(self, x):
        x_out = self.channel_attention(x) # 입력 데이터 x에 대해 채널 어텐션 적용 후 x_out에 저장
        # if self.spatial: # spatial=True 이면
        x_out = self.spatial_attention(x_out) # 위의 x_out에 Spatial_Attention 적용

        return x_out



##### resnet50 정의
class BottleNeck(nn.Module):      ###### BottleNeck 정의

    # def __init__(self, in_channels, out_channels, expansion=4, stride=1, use_cbam=True):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1):

        super(BottleNeck, self).__init__()

        # self.use_cbam = use_cbam
        # 첫 컨볼루션 레이어만 stride=2고 나머지는 다 stride=1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

        # identity_connection(skip connection=잔차모듈) 매핑 시 입출력의 크기가 같아야 하므로 stride나 out_channels(출력 채널)의 수가 각각 1과 expansion*out_channels보다 클 때마다 입력 x도 동일한 크기로 다운샘플링 돼야함
        self.identity_connection = nn.Sequential()
        if stride != 1 or in_channels != expansion*out_channels:
            self.identity_connection = nn.Sequential(  # self.identity_connection에 1x1 컨볼루션 레이어와 배치 정규화 레이어를 추가
                nn.Conv2d(in_channels=in_channels, out_channels=expansion*out_channels, kernel_size=1, stride=stride, bias=False), # stride에 따라 입력 x의 크기가 조절됨
                nn.BatchNorm2d(num_features=out_channels*expansion)
            )   # x의 채널 수를 expansion*out_channels와 같게 맞추고, 스트라이드 값이 1이 아닌 경우에는 입력 x의 크기를 동일하게 맞춰줌

        # if self.use_cbam:
        # self.cbam = CBAM(channel_in=out_channels*expansion)


    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # if self.use_cbam:
        # out = self.cbam(out)

        out += self.identity_connection(x) #identity connection/skip connection  ## 잔차모듈 사용위해 self.identity_connection에 x적용
        out = self.relu(out)

        return out


class ResNet50(nn.Module):

    # def __init__(self, use_cbam=True, image_depth=3, num_classes=6):
    # def __init__(self, use_cbam=True, image_depth=3, num_classes=2):
    def __init__(self, image_depth=3, num_classes=2):
        super(ResNet50, self).__init__()

        self.in_channels = 64
        self.expansion = 4
        self.num_blocks = [3, 3, 3, 2]  # 각 레이어의 bottleneck block 수

        self.conv_block1 = nn.Sequential(nn.Conv2d(kernel_size=7, stride=2, in_channels=image_depth, out_channels=self.in_channels, padding=3, bias=False),
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(stride=2, kernel_size=3, padding=1))

        self.layer1 = self.make_layer(out_channels=64, num_blocks=self.num_blocks[0], stride=1)
        self.layer2 = self.make_layer(out_channels=128, num_blocks=self.num_blocks[1], stride=2)
        self.layer3 = self.make_layer(out_channels=256, num_blocks=self.num_blocks[2], stride=2)
        self.layer4 = self.make_layer(out_channels=512, num_blocks=self.num_blocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, padding=3)  # img_size: (512, 512) 일 때
        # self.avgpool = nn.AvgPool2d(7)  # img_size: (512, 512) 일 때
        # self.linear = nn.Linear(512*self.expansion*9, num_classes) # img_size: (512, 512) 일 때

        self.avgpool = nn.AvgPool2d(7)  # img_size: (640, 360) 일 때

        self.linear = nn.Linear(512*self.expansion*2, num_classes)  # img_size: (640, 360) 일 때
        # self.sigmoid = nn.Sigmoid()
        # self.linear = nn.Linear(512*self.expansion, num_classes-1)

        # print(self.linear)
        # print('-----------------------------------')


    def make_layer(self, out_channels, num_blocks, stride): # bottleneck layer 생성

        strides = [stride] + [1]*(num_blocks-1)  # 병목 블록들 간의 입력 데이터의 공간적 차원의 다운샘플링을 제어하는 역할
        # bottleneck은 1x1, 3x3, 1x1 conv 연산이기 때문에 3x3 conv연산의 스트라이들을 통해 입력 데이터의 공간적 차원의 다운샘플링을 제어. bottleneck의 첫 conv 연산(self.conv1)의 stried 값이 stride로
        # 되어있기 때문에(make_layer의 인자인 stride가 입력됨) 첫 번째 병목 블록의 스트라이드를 stride로 설정하고, 나머지 병목 블록들의 스트라이드를 1로 설정함(나머지는 stride 설정이 없어서 default임)
        layers = []  # 생성된 병목 블록들을 저장하기 위한 리스트: 병목 블록은 bottleneck 클래스를 인스턴스화하여 생성
        for stride in strides:
            layers.append(BottleNeck(in_channels=self.in_channels, out_channels=out_channels, stride=stride, expansion=self.expansion))
            # bottleneck 클래스의 인스턴스를 만들어서 layer 리스트에 추가
            self.in_channels = out_channels * self.expansion # in_channels, out_channels 업데이트
        return nn.Sequential(*layers)  # nn.Sequential 클래스 사용해 layers 리스트에 있는 모든 bottleneck 인스턴스를 순차적으로 실행할 수 있는 모듈 생성.
    # *layers는 리스트의 요소들을 언패킹하여 전달하는 구문으로, nn.Sequential에게 layers 리스트에 있는 모든 BottleNeck 인스턴스들을 전달하여 순차적으로 실행할 수 있는 모듈을 생성


    def forward(self, x):

        x = self.conv_block1(x)
        x = self.layer1(x)  # bottleneck layer
        x = self.layer2(x)  # bottleneck layer
        x = self.layer3(x)  # bottleneck layer
        x_conv = self.layer4(x)  # bottleneck layer
        x = self.avgpool(x_conv)
        x = nn.Flatten()(x) #flatten the feature maps.
        x = self.linear(x)
        # x = self.sigmoid(x) #원래 없음

        return x_conv, x


###### CustomDataset 정의
class CustomDataset(Dataset):

    def __init__(self, dataset_folder_path=args["data_folder"], image_size=64, image_depth=3, train=True, transform=None):
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform  # HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()

    def __len__(self):
        return len(self.image_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, label = self.image_path_label[idx]

        if self.image_depth == 1:
            image = cv2.imread(image, 0)
        else:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)

        if self.transform:
            image = self.transform(image)

        # return image, label
        return {'image': image, 'label': label}  # 딕셔너리 형태로 이미지 경로, 해당 이미지의 클래스의 인덱스를 저장


    def get_classnames(self):
        return os.listdir(f"{self.dataset_folder_path}train/")  # violence, nonviolence

    def read_folder(self):
        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path}train/"
        else:
            folder_path = f"{self.dataset_folder_path}val/"  # valid도 가능

        for x in glob.glob(folder_path + "**", recursive=True):  # 해당 폴더의 하위 폴더까지 탐색
            if not x.endswith('png'):
                continue

            class_idx = self.classes.index(x.split('\\')[-2])  # 클래스 이름(Violence, Nonviolence)의 인덱스 저장 # ubi, ucfcrime
            image_path_label.append(
                (x, int(class_idx)))  # (D:/InChang/SCVD/frames_new_split_train/train\NonViolence\nv59\nv59-0223.jpg, 0)
        return image_path_label  # (이미지 경로, 클래스 인덱스)

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path=f"{model_save_folder}3090ubi640x360_image_modelearly.pth"):
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

##### Train
def train(args, device="cuda:0"):
    model = ResNet50(image_depth=args["img_depth"], num_classes=args["num_class"])
    model.to(device)

    optimizer = Adam(model.parameters(), args["learning_rate"])
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args["decay_rate"])
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # summary(model, (3, 64, 64))
    summary(model, (3, 640, 360))

    ##### train, test dataset 정의
    train_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                                  image_depth=args["img_depth"],
                                  train=True,
                                  transform=transforms.ToTensor())  # transforms.ToTensor(): HxWxC -> CxHxW, 이미지 픽셀 밝기 값은 0~255의 범위에서 0~1의 범위로 변경
    test_dataset = CustomDataset(dataset_folder_path=args["data_folder"], image_size=args["img_size"],
                                 image_depth=args["img_depth"],
                                 train=False, transform=transforms.ToTensor())

    train_generator = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"],
                                 pin_memory=True)  # , sampler=sampler
    test_generator = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"],
                                pin_memory=True)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    best_accuracy = 0

    for epoch in range(args["epoch"]):
        model.train()

        loss_per_epoch = []
        accuracy_per_epoch = []
        i = 0
        total = 0
        batch_accuracy_ = 0
        y_true = []
        y_pred = []

        for i, train_data in tqdm(enumerate(train_generator)):

            x_train, y_train = train_data['image'].to(device, non_blocking=True), train_data['label'].to(device, non_blocking=True)

            optimizer.zero_grad()

            _, output = model(x_train)
            # train_loss = criterion(input=output.to(torch.float32), target=y_train.to(torch.float32)) # bce loss
            train_loss = criterion(input=output, target=y_train)

            train_loss.backward()
            optimizer.step()

            num_data = y_train.size()[0]
            preds = torch.argmax(output, dim=1)
            correct_pred = torch.sum(preds == y_train)
            batch_accuracy = correct_pred * (100/num_data)

            _, predicted = torch.max(output, 1)
            total += y_train.size(0)
            batch_accuracy_ += (predicted == y_train).sum().item()

            loss_per_epoch.append(train_loss.item())
            accuracy_per_epoch.append(batch_accuracy.item())

        curr_train_accuracy = sum(accuracy_per_epoch) / (i + 1)
        curr_train_loss = sum(loss_per_epoch) / (i + 1)

        train_loss_list.append(curr_train_loss)
        train_acc_list.append(curr_train_loss)

        print(f"Epoch {epoch + 1}/{args['epoch']}")
        print('train loss: ', curr_train_loss, '  ', 'train accuracy: ', curr_train_accuracy)

        model.eval()
        loss_per_epoch = []
        accuracy_per_epoch = []
        i = 0
        total = 0
        batch_accuracy_ = 0

        with torch.no_grad():
            for i, test_data in tqdm(enumerate(test_generator)):
                x_test, y_test = test_data['image'].to(device, non_blocking=True), test_data['label'].to(device, non_blocking=True)

                _, output = model(x_test)

                # test_loss = criterion(input=output.to(torch.float32), target=y_test.to(torch.float32)) # bceloss
                test_loss = criterion(input=output, target=y_test) # crods entropy loss

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

                y_pred.append(output.detach().cpu().numpy())
                y_true.append(y_test.detach().cpu().numpy())

            curr_test_accuracy = sum(accuracy_per_epoch) / (i + 1)
            curr_test_loss = sum(loss_per_epoch) / (i + 1)

            test_loss_list.append(curr_test_loss)
            test_acc_list.append(curr_test_accuracy)

        print('validation loss: ', curr_test_loss, '  ', 'validation accuracy: ', curr_test_accuracy)
        if epoch % 5 == 0:
            lr_decay.step()
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}")

        if best_accuracy < curr_test_accuracy:
            torch.save(model.state_dict(), f"{model_save_folder}/3090ubi640x360_image_model.pth")
            best_accuracy = curr_test_accuracy
            print('Current model has best valid accuracy')

        print('\n--------------------------------------------------------------------------------\n')

    y_pred = np.array(y_pred, dtype=np.int64)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)  # .max(axis=1)

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

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
    plt.savefig(args["graphs_folder"]+f"AUC_graph051zz7.png")
    plt.show()

    print('AUC: ', roc_auc)

    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.title('Train and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"]+f"rwf2000_acc_graphoriubi0517.png")
    plt.show()

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('Train and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([1, args["epoch"]])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"]+f"rwf2000_loss_graphoriubi0517.png")
    plt.show()


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    train(args=args, device=device)

    ### 지금 이게 아나콘다 파워쉘 프롬프트 마지막꺼에서 돌아가고 있고 ubi-fights 640,360 데이터 이미지로
    # fight가 0 normal이 1