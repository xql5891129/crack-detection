说明文档

1. **运行环境**

   PC：windows11、RTX4060

   Python：3.12

   Pytorch：2.3.1

   cuda：12.6

依赖库参照项目内 requirements.txt

1. **文件结构**

   CrackForest-dataset-master：数据集

   inputs/data：test：测试集  train:训练集

   json：自己标注的图片分割数据

   models：log.csv（训练日志）、 model.pth（模型） 、config.yml（训练参数）

   outputs：测试集输出图片

   ResNet50：resnet权重文件

   Dataset.py：处理数据集

   jsonTopng.py：将图片json格式转为png格式

   losses.py：自定义损失函数

   metrics.py、utils.py：模型评估函数

   Resnet.py：resnet模型结构定义

   Unet.py：unet结构定义

   Model.py：模型结构定义

   resize.py：改变图片大小

   Test.py：测试

   train.py：模型训练


