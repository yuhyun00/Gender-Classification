# Image-classification---emotion
## Description
#### Crop된 얼굴의 감정을 판단하는 네트워크를 개발하는 Task

## DB 및 backbone
#### Image Detection
#### https://www.kaggle.com/cashutosh/gender-classification-dataset/code
#### backbone code - ResNet
####  https://github.com/Jongchan/week1_tiny_imagenet_tutorial

## Result

#### VGGNet
1. vgg11_m4
2. batch_size = 512
3. learning_rate = 0.1
4. lr_drop_epochs: [10, 15]
5. learning_drop_rate = 0.1
6. epoch = 20
7. transforms.Compose([transforms.Resize( (96,96)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.18, 0.1 8, 0.18))])
8. val_acc = 97.36

#### ResNet
1. resnet18
2. batch_size = 512
3. learning_rate = 0.01
4. lr_drop_epochs: [10, 15]
5. learning_drop_rate = 0.1
6. epoch = 20
7. transforms.Compose([transforms.Resize((96, 96)), transforms.RandomPerspective(), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.18,0.18,0.18))])
8. val_acc = 97.33
