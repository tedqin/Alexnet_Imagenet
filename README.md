#### -本科大创项目，留存纪念

### 依赖
* python 2.7
* tensorflow gpu 1.10
* numpy
* opencv

### 模型结构
Alexnet有5个卷积层和3个全连接层，并且Alex发现移除任意一层都会降低最终的效果。网络结构如图

![](https://i.imgur.com/0LFUfXp.png)

这个网络前面5层是卷积层，后面三层是全连接层，最终softmax输出是1000类。

### 数据集
![](https://i.imgur.com/qm2Ofif.png)

### 测试
caffe提供的预训练模型一共包括了1000类图像，涵盖了生物、天文、自然、科技等多个方面的相关图像，具体在caffe_classes.py下，由于文件本身较大就没放在github上，可以很容易在网上找到  
测试集包括1000张图像，共200类，标签存储在lable.txt当中。取测试图像概率最大类的下标作为测试的分类结果（即top-1测试），并且与label对比，最后计算得出整个测试集得precision在75%左右，算是比较好的结果。  
![](https://i.imgur.com/SOuRIsz.png)
![](https://i.imgur.com/pWERo64.png)

### 总结
AlexNet可以说是深度神经网络的鼻祖，相比于后来的VGG和googlenet而言，它的构造更加简单，而且结构也已经非常的成熟和稳固。  