# Keras-segnet
keras库实现segnet语意分割模型

# 配置
本模型的数据采用segnet作者所使用的数据集，可以到[SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)这里下载，将CamVid目录下的test/testannot/train/trainannot/val/valannot六个文件拷贝到此项目的CamVid目录。

# 数据集介绍
该数据集分12个类，包括背景。其中样本数据存放在test/train/val三个目录下，样本标签存放在testannot/trainannot/valannot三个目录中

# 模型
本项目使用VGG16在imagenet上的权重作为迁移学习，训练时需要下载vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5模型权重，并放到项目的models目录。

# 训练
执行python train.py开始训练，训练每三个epochs保存一次权重，保存到项目logs目录下。
