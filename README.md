# DeepLearning-Playground
 Some recreation of learing  
 小孩子不懂事做着玩的

## #1 AbstractIMG
Classify sentiment of abstract image   
构建知识蒸馏模型  KD.py  
VGG使用google提供的*vgg19_weights_tf_dim_ordering_tf_kernels.h5*  
收到需求添加领域自适应，使用DSN  
参考论文
> Bousmalis K, Trigeorgis G, Silberman N, Krishnan D, Erhan D. Domain separation networks. Advances in neural information processing systems. 2016;29.

## #2 recommend
A simple CTR model  
特征构建后简单使用DeepFM。评论作为补充，编码后输入网络

## #3 FOCAL_LOSS.py
3 implementations of focal loss  
参考论文
>- Lin TY, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. InProceedings of the IEEE international conference on computer vision 2017 (pp. 2980-2988).
>- 崔子越, et al."结合改进VGGNet和Focal Loss的人脸表情识别." 计算机工程与应用 57.19(2021):171-178
>- 傅博文.基于CNN的图像情感分析研究.2020.杭州电子科技大学