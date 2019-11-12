---
title: 深度学习计算机视觉实战(二) ：pytorch上实现对小数据集的迁移学习Transfer Learning
date: 2019-04-22 12:00:00
categories: Learning Note
description: 本文从概念出发，讨论迁移学习在不同场景下的使用方法，并通过实例在pytorch上运行来进一步巩固。
tags: [Computer Vision, Transfer Learning, pytorch]
cover: img/blogs/3.jpg
---

# 前言
本文从概念出发，讨论迁移学习在不同场景下的使用方法，并通过实例在pytorch上运行来进一步巩固。

# 迁移学习(Transfer Learning)

## 概念
传统的机器学习/数据挖掘只有在训练集数据和测试集数据都来自同一个feature space（特征空间）和统一分布的时候才运行的比较好，这意味着每一次换了数据都要重新训练模型，以至于过于麻烦。对于绝大多数开发者来说，在实践中很少有人从头开始训练整个卷积网络（使用随机初始化），因为拥有足够大小的数据集相对较少。即便是不同的事物，feature space都有一定的共通性，这使得在训练过的成熟网络同样通过微调、部分修改后适用于相近的一系列问题。故迁移学习是十分高效的一种学习方式。
在实践中，由于数据集不够大，很少有人从头开始训练网络。常见的做法是使用预训练的网络来重新fine-tune，或者当做特征提取器。常见的两类迁移学习场景：
1. **卷积网络当做特征提取器**。使用在ImageNet上预训练的网络，去掉最后的全连接层，剩余部分当做特征提取器（例如AlexNet在最后分类器前，是4096维的特征向量）。这样提取的特征叫做CNN codes。得到这样的特征后，可以使用线性分类器（Liner SVM、Softmax等）来分类图像。
2. **Fine-tuning卷积网络**。替换掉网络的输入层（数据），使用新的数据继续训练。Fine-tune时可以选择fine-tune的全部层或部分层。通常，前面的层提取的是图像的通用特征（generic features）（如边缘检测，色彩检测），这些特征对许多任务都有用。后面的层提取的是与特定类别有关的特征，故fine-tune时常常只需要采用较小的学习率来Fine-tuning后面的层。

## 应用场景
**使用迁移学习时要注意，预训练模型应用场景，要和当前任务差距不大**
1. **待训练的数据集较小，已训练的模型和当前任务相似**。只重新训练已有模型的靠近输出的几层，例如将ImageNet中输出层原来可以判别一万种输出的网络改为只判别猫的品种，从而利用已有网络来做低层次的特征提取。
2. **待训练的数据集较小，已训练的模型和当前任务场景差距较大**。例如你有的已训练网络能识别出白天高速路上的违章车辆，你需要训练一个能识别出夜间违章车辆的模型，由于不管白天夜晚，交通规则是没有变化的。故只需将网络靠近输入的那几层重新训练，等到新的网络能够提取出夜间车辆的基本信息后，就可用预训练模型，而不是从头开始。
3. **待训练的数据集较大，已有模型的训练数据和现有的训练数据类似**。使用原网络的结构，并保留每一层的节点权重，再逐层微调。
4. **待训练的数据集较大，已有的模型和新模型的数据差异度很高**。从头开始，重新训练。
> 参考:CS231n课程

# Pytorch代码详解

## 开发环境
工具包 | 版本
-------- | -----
python | 3.6.2
torch | 1.0.0
torchvision | 0.2.1
visdom | 0.1.7
matplotlib | 2.0.2
numpy | 1.15.4

## 数据集说明
下载链接：https://download.pytorch.org/tutorial/hymenoptera_data.zip
这是pytorch官方的数据集，取自于imageNet的非常小的子集。
其训练集和验证集的数目见下表：

类别 | 训练集 | 验证集
--------|--------|--------
 蜜蜂|121|83
 蚂蚁|124|70

 由于数据集十分有限，我们无法从零开始训练成有较好效果的网络，但迁移学习+一些正则化方法(如数据增强)的引入，使得其成为了可能。符合应用场景1：**待训练的数据集较小，已训练的模型和当前任务相似**，故我们只需**重新训练已有模型的靠近输出的几层**即可。

## 代码详解

### 库依赖
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy 	# 浅拷贝 copy 深拷贝 deepcopy
```

### 数据集加载
```python
######################################################################
# 数据加载
# ---------
# 训练集：数据增强+标准化
# 验证集：标准化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 将图片随机裁切成224*224的大小
        transforms.RandomHorizontalFlip(), # 将图片进行随机水平翻转
        transforms.ToTensor(), # 将图片转换成tensor数据格式
        # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray转换成
        # 形状为[C,H,W]取值范围是[0,1.0]的torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 给定均值：(R,G,B) 方差：(R,G,B),将Tensor正则化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), # 将图片随机裁切成256*256的大小
        transforms.CenterCrop(224), # 图片中心裁切成224*224的大小
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data' # 数据集路径名
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                # ImageFolder : 通用数据加载器，根据文件夹的名称给其分类
                #  para1: 根文件夹路径train/val  join方法:路径拼接
                #  para2: 调用data_transforms的图片预处理
                  	for x in ['train', 'val']}
dataloaders = { x: torch.utils.data.DataLoader(image_datasets[x],
				# 数据集组合采样器，并在数据集上提供单进程或多进程迭代器
                batch_size=16, 	# mini-batch的每个batch包含的数据大小
                shuffle=True, 	# 随机打乱
                num_workers=0) 	# 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
                for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # 记录train/var的数据集大小
class_names = image_datasets['train'].classes # 数据的类型名
```

**torchvision.datasets.ImageFolder**：一个通用的数据加载器，数据集中的数据以以下方式组织
> root/dog/xxx.png
> root/dog/xxx.png
> root/dog/xxx.png
> ...
> root/cat/123.png
> root/cat/nsdf3.png
> root/dog/xxz.png
> ...
```python
dset.ImageFolder(root="root folder path", [transform, target_transform])
```
他有以下成员变量:
> self.classes - 用一个list保存 类名
>  self.class_to_idx - 类名对应的 索引
> self.imgs - 保存(img-path, class) tuple的list

### 图片数据显示
```python
######################################################################
# 图片数据显示
# ----------------------
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) 	# 三维转置,交换维度, (C,H,W)->(H,W,C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean					# 归一化
    inp = np.clip(inp, 0, 1) 				# 将矩阵中的元素限制在a_min, a_max之间, 标准化
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  						# 暂停一段时间使得plot得以正常更新


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))  
	# 通过iter()函数获取这些可迭代对象的迭代器,对获取到的迭代器不断使用
	# next()函数来获取下一条数据
	# (iter()函数实际上就是调用了可迭代对象的__iter__方法)
	# inputs为一个batch的图片集合 (B x C x H x W) 
# 制作图片阵列
out = torchvision.utils.make_grid(inputs) # 类似于做图像拼接，横向摆放方便显示(自带padding)
torchvision.utils.save_image(inputs, 'inputInstances.JPG') # 图像保存
imshow(out, title=[class_names[x] for x in classes])	   # 图像展示
```

### 模型训练函数
```python
######################################################################
# 模型训练
# ------------------
# -  Scheduling the learning rate
# -  Saving the best model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()		# 计时

    best_model_wts = copy.deepcopy(model.state_dict())  # state_dict: 返回包含模块整个状态的字典
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 对每一个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 对数据进行迭代
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()		# 参数梯度归零

                # with 语句：适用于对资源进行访问的场合，确保不管使用过程中是否发生异常
                # 都会执行必要的“清理”操作，释放资源
                with torch.set_grad_enabled(phase == 'train'):  # 根据其参数模式启用或禁用grads
                	# forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 返回给定维度dim中输入张量的每一行的最大值,即预测值
                    loss = criterion(outputs, labels) # loss值计算

                    # backward + 仅在训练阶段进行参数更新
                    if phase == 'train':
                        loss.backward()			# 计算当前张量的梯度并逆向传播更新参数
                        optimizer.step()		# 执行单个优化步骤

                # information statistics
                running_loss += loss.item() * inputs.size(0) 	# item() → number
                												# size(0): size结果的第一位
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
          		# deep copy the model
            	# 深度复制：完全复制整个模型，修改新模型不会影响原模型
        print() # \n

    time_elapsed = time.time() - since # 计算总用时
    print('Training complete in {:.0f}m {:.0f}s'
    	.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型的参数/权重
    model.load_state_dict(best_model_wts)
    return model
```

### 预测效果展示函数
```python
######################################################################
# 模型预测效果显示函数
# ---------------------------------
# Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()      # Sets the module in evaluation mode.
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)	# 预测

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)       # 将多个图画到一个平面
                ax.axis('off')		# 不显示坐标轴
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)          # Sets the module in training mode
                    return
        model.train(mode=was_training)
```

### 方案一：参数微调
```python
#############################################################################################
# 卷积网络参数微调
# ----------------------
# 加载训练好的resnet18的模型，并修改最后一层使得成为二分类问题
model_ft = models.resnet18(pretrained=True)	# 加载训练好的resnet18的模型
num_ftrs = model_ft.fc.in_features  		# in_features: 返回全连接层的输入尺寸
model_ft.fc = nn.Linear(num_ftrs, 2)		# 修改全连接层，使得最终输出尺寸为2

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()	# 交叉熵损失函数

# 使用SGD进行参数更新， lr设低，微调即可
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 学习率衰减   Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) 

#############################
# 训练+验证
# ------------------
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

visualize_model(model_ft)
torch.save(model_ft, 'model_Finetuning.pkl')
```

### 方案二：固定特征提取器
```python
############################################################################################
# ConvNet作为固定特征提取器
# ----------------------------------
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.

model_conv = torchvision.models.resnet18(pretrained=True)	# 加载训练好的resnet18的模型
for param in model_conv.parameters():  	# 冻结参数使得在backward时不会更新参数
	param.requires_grad = False

# 默认情况下，新构造的模块的参数具有requires_grad = True
# 使得在训练时只更新最后全连接层的参数
num_ftrs = model_conv.fc.in_features	# 同finetuning
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#############################
# 训练+验证
# ------------------
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)
torch.save(model_conv, 'model_Fixed.pkl')
```

## 运行结果

**方案一：参数微调**

epoch | 训练集损失值 | 训练集准确度(%) | 验证集损失值| 验证集准确度(%)
-------- | -------- | ----------- | ----------- | -----------
1 |0.5876|66.80|0.4734|72.55
5 |0.2054|90.98|0.1782|94.12
10|0.1296|94.67|0.1325|96.73
15|0.1835|92.62|0.1296|96.73
20|0.1232|94.26|0.1301|96.08
25|0.1864|93.64|0.1273|**96.73**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190422212120393.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0NTkzMQ==,size_16,color_FFFFFF,t_70#pic_center)

**方案二：固定特征提取器**

epoch | 训练集损失值 | 训练集准确度(%) | 验证集损失值| 验证集准确度(%)
-------- | -------- | ----------- | ----------- | -----------
1 |0.6379|63.93|0.4140|90.20
5 |0.2322|91.39|0.1953|92.81
10|0.1927|93.03|0.1811|94.77
15|0.2115|93.03|0.1794|95.42
20|0.1892|94.26|0.1772|**96.08**
25|0.2136|91.39|0.1755|95.42

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190422212130664.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0NTkzMQ==,size_16,color_FFFFFF,t_70#pic_center)

## 完整训练源码
***github链接***
> https://github.com/yoooooohu/Transfer_Learning_Official
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy 	# 浅拷贝 copy 深拷贝 deepcopy

plt.ion()   # 开启plt交互模式

######################################################################
# 数据加载
# ---------
# 训练集：数据增强+归一化
# 验证集：归一化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),	# 将图片随机裁切成224*224的大小
        transforms.RandomHorizontalFlip(),	# 将图片进行随机水平翻转
        transforms.ToTensor(),				# 将图片转换成tensor数据格式
        # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray转换成
        #   形状为[C,H,W]取值范围是[0,1.0]的torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        									# 给定均值：(R,G,B) 方差：(R,G,B),将Tensor归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),				# 将图片随机裁切成256*256的大小
        transforms.CenterCrop(224),			# 图片中心裁切成224*224的大小
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'				# 数据集路径名
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
							# ImageFolder : 通用数据加载器，根据文件夹的名称给其分类
								# para1: 根文件夹路径train/val  join方法:路径拼接
                                # para2: 调用data_transforms的图片预处理
                  	for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
				# 数据集组合采样器，并在数据集上提供单进程或多进程迭代器
							batch_size=16, 	# mini-batch的每个batch包含的数据大小
							shuffle=True, 	# 随机打乱
							num_workers=0) 	# 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
              		for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}	# 记录train/var的数据集大小
class_names = image_datasets['train'].classes # 数据的类型名

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# 图片数据显示
# ----------------------
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) 	# 三维转置,交换维度, (C,H,W)->(H,W,C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean					# 归一化
    inp = np.clip(inp, 0, 1) 				# 将矩阵中的元素限制在a_min, a_max之间, 标准化
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  						# 暂停一段时间使得plot得以正常更新


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))  
	# 通过iter()函数获取这些可迭代对象的迭代器,对获取到的迭代器不断使用
	# next()函数来获取下一条数据
	# (iter()函数实际上就是调用了可迭代对象的__iter__方法)
	# inputs为一个batch的图片集合 (B x C x H x W) 
# 制作图片阵列
out = torchvision.utils.make_grid(inputs) # 类似于做图像拼接，横向摆放方便显示(自带padding)
torchvision.utils.save_image(inputs, 'inputInstances.JPG') # 图像保存
imshow(out, title=[class_names[x] for x in classes])	   # 图像展示

######################################################################
# 模型训练
# ------------------
# -  Scheduling the learning rate
# -  Saving the best model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()		# 计时

    best_model_wts = copy.deepcopy(model.state_dict())  # state_dict: 返回包含模块整个状态的字典
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 对每一个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 对数据进行迭代
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()		# 参数梯度归零

                # with 语句：适用于对资源进行访问的场合，确保不管使用过程中是否发生异常
                # 都会执行必要的“清理”操作，释放资源
                with torch.set_grad_enabled(phase == 'train'):  # 根据其参数模式启用或禁用grads
                	# forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 返回给定维度dim中输入张量的每一行的最大值,即预测值
                    loss = criterion(outputs, labels) # loss值计算

                    # backward + 仅在训练阶段进行参数更新
                    if phase == 'train':
                        loss.backward()			# 计算当前张量的梯度并逆向传播更新参数
                        optimizer.step()		# 执行单个优化步骤

                # information statistics
                running_loss += loss.item() * inputs.size(0) 	# item() → number
                												# size(0): size结果的第一位
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
          		# deep copy the model
            	# 深度复制：完全复制整个模型，修改新模型不会影响原模型
        print() # \n

    time_elapsed = time.time() - since # 计算总用时
    print('Training complete in {:.0f}m {:.0f}s'
    	.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型的参数/权重
    model.load_state_dict(best_model_wts)
    return model

######################################################################
# 模型预测效果显示函数
# ---------------------------------
# Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()      # Sets the module in evaluation mode.
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)	# 预测

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)       # 将多个图画到一个平面
                ax.axis('off')		# 不显示坐标轴
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)          # Sets the module in training mode
                    return
        model.train(mode=was_training)

#############################################################################################
# 卷积网络参数微调
# ----------------------
# 加载训练好的resnet18的模型，并修改最后一层使得成为二分类问题
model_ft = models.resnet18(pretrained=True)	# 加载训练好的resnet18的模型
num_ftrs = model_ft.fc.in_features  		# in_features: 返回全连接层的输入尺寸
model_ft.fc = nn.Linear(num_ftrs, 2)		# 修改全连接层，使得最终输出尺寸为2

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()	# 交叉熵损失函数

# 使用SGD进行参数更新， lr设低，微调即可
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 学习率衰减   Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) 

#############################
# 训练+验证
# ------------------
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

visualize_model(model_ft)
torch.save(model_ft, 'model_Finetuning.pkl')

#############################################################################################
# ConvNet作为固定特征提取器
# ----------------------------------
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.

model_conv = torchvision.models.resnet18(pretrained=True)	# 加载训练好的resnet18的模型
for param in model_conv.parameters():  	# 冻结参数使得在backward时不会更新参数
	param.requires_grad = False

# 默认情况下，新构造的模块的参数具有requires_grad = True
# 使得在训练时只更新最后全连接层的参数
num_ftrs = model_conv.fc.in_features	# 同finetuning
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#############################
# 训练+验证
# ------------------
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)
torch.save(model_conv, 'model_Fixed.pkl')

plt.ioff()
plt.show()
```

# 修改记录
Time | Note | Author
-------- | ----- | -------
19.4.18 | 原始版本 | Yooo_Hu
19.4.22 |注释修改 | Yooo_Hu