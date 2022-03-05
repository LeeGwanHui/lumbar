## 라이브러리 추가하기
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from util import *
from dataset import *
import matplotlib.pyplot as plt

from torchvision import transforms

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=32, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=5, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log_256", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="on", type=str, dest="train_continue")
parser.add_argument("--port", default=52162)

args = parser.parse_args()

## 트레이닝 파라메터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), Resize(),RandomFlip(),RandomRotation(),ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory = True)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory = True)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else: # test
    transform = transforms.Compose([Normalization_test(mean=0.5, std=0.5), Resize_test(), ToTensor_test()])

    dataset_test = Dataset_test(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory = True)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
# fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_loss = nn.MultiLabelSoftMarginLoss().to(device) # 다중 클래스 이기 때문에
## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기 output 저장을 위한 함수
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) #from tensor to numpy
fn_denorm = lambda x, mean, std: (x * std) + mean  #denormalize
fn_class = lambda x: 1.0 * (x > 0.5) #classification() using thresholding( p=0.5) 네트워크 output을 binary class로 변환

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr=[]

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device=device)
            input = data['input'].to(device=device, dtype=torch.float32)
            output = net(input)
            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            # label = fn_tonumpy(label)
            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            # output = fn_tonumpy(fn_class(output))
            #
            # writer_train.add_image('label', label[..., 0:1], num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output[..., 0:1], num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            #
            # writer_train.add_image('label_back', label[..., 6:7], num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output_back', output[..., 6:7], num_batch_train * (epoch - 1) + batch, dataformats='NHWC')


        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device=device)
                input = data['input'].to(device=device, dtype=torch.float32)
                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                # label = fn_tonumpy(label)
                # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                # output = fn_tonumpy(fn_class(output))

                # writer_val.add_image('label', label[..., 0:1], num_batch_train * (epoch - 1) + batch,dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output[..., 0:1], num_batch_train * (epoch - 1) + batch,dataformats='NHWC')
                #
                # writer_val.add_image('label_back', label[..., 6:7], num_batch_train * (epoch - 1) + batch,dataformats='NHWC')
                # writer_val.add_image('output_back', output[..., 6:7], num_batch_train * (epoch - 1) + batch,dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    print("epoch: %.4e" % st_epoch)

    shape_lst = []
    dir_save_test = os.path.join(data_dir, 'test')
    lst_data = os.listdir(dir_save_test)

    for i in range(len(lst_data)):
        k = np.load(os.path.join(dir_save_test, 'input_%03d.npy' % i))
        revered_list = list(reversed(k.shape))
        b = tuple(revered_list)
        shape_lst.append(b)

    with torch.no_grad():
        net.eval()

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            input = data['input'].to(device)
            output = net(input)

            print("TEST: BATCH %04d / %04d |" % (batch, num_batch_test))

            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(input.shape[0]):
                id = batch_size * (batch - 1) + j

                output_new = []
                # 이미지 resize
                for ii in range(output.shape[-1]):
                    ex_new = cv2.resize(output[j][..., ii: ii+1], dsize=shape_lst[id], interpolation=cv2.INTER_NEAREST)
                    output_new.append(ex_new) # list
                a = np.array(output_new) # numpy array로 변경
                b = a.transpose(1, 2, 0) # 축이 channel x row x column -> row x column x channel

                plt.imsave(os.path.join(result_dir, 'png', 'input_%03d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%03d.png' % id), b[...,6:7].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'output_%03d.npy' % id), b.astype(np.uint8))