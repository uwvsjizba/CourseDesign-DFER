# CourseDesign-DFER 
———— 基于卷积神经网络识别面部表情的机器学习课程设计

#### 本项目
- 选择 VGG19 模型以及 Resnet18 模型作为训练模型 
- 选择 FER2013 和 CK+ 作为数据集

#### 用两种数据集训练VGG19模型
- python mainpro_CK+.py --model VGG19 --bs 10 --lr 0.01 (以 CK+ 数据集训练 VGG19 模型)
- python mainpro_FER.py --model VGG19 --bs 10 --lr 0.01 (以 FER2013 数据集训练 VGG19 模型)

#### 用两种数据集训练Resnet18模型
- python mainpro_CK+.py --model Resnet18 --bs 10 --lr 0.01 (以 CK+ 数据集训练 Resnet18 模型)
- python mainpro_FER.py --model Resnet18 --bs 10 --lr 0.01 (以 FER2013 数据集训练 Resnet18 模型)

#### 结果
以最终的训练效果来对比两个模型的精确性并总结优缺点

#### 参考信息来源：
- 吴捷大佬的博客地址: https://zhuanlan.zhihu.com/p/39779767
- 吴捷大佬的论文地址: https://arxiv.org/pdf/1811.04544