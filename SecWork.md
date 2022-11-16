### Introduction

视觉情感分析的大多数研究集中在对于情感图像的标签多分类问题上。然而图像中蕴含了丰富情感，单标签分类 多标签分类无法满足对于一副图像中情感的细致描述需求。作为一种更加泛化的学习范式，标签分布学习成为了情感分析领域的研究热点。



一些工作尝试使用心理学领域对于情感的研究结果作为先验知识[JIEC][]，然而





### Problem Formulation

耿鑫的LDL综述或ppt

Learning Expectation of Label Distribution for Facial Age and Attractiveness Estimation



First of all, the description degree $d_{x}^{y}$ could be represented by the form of conditional probability, i.e., dy x= P(y|x). This might be explained as that given an instance $x$, the probability of the presence of $y$ is equal to its description degree. Then, the problem of label distribution learning can be formulated as follows.

Let ${\cal X} = \mathbb{R}^{q}$ denote the input space and $ {\cal Y} = {y_1,y_2,··· , y_c} $ denote the finite set of possible class labels. Given a training set $ S = {(x_1,D_1), (x_2,D_2),··· ,(x_n,D_n)} $, where {x_i}∈ X is an instance, $ Di= {dy1 xi,dy2 xi,··· ,dyc xi} $ is the label distribution associated with ${x_i}$, the goal of label distribution learning is to learn a conditional probability mass function $p(y|x)$ from S, where $x \in {\cal X} $and $y \in {\cal Y}$



### Methods

MAGNET_ Multi-Label Text Classification using Attention-based Graph__Neural Network

Zero-Shot Learning via Category-Specific Visual-Semantic Mapping and Label Refinement

![image-20201023194640989](C:\Users\15244\AppData\Roaming\Typora\typora-user-images\image-20201023194640989.png)

**要表达的点：**

分类不如分布

分布需要考虑情感标签相关性，但图像信息丰富 标签的出现并不符合先验知识

Exploring Correlation between Labels to improve Multi-Label Classification

相似核没有被用于关注局部信息，而是被用于发现元素之间的关系

不使用Softmax的原因

不使用KL散度（不对称 且未考虑发生差异的位置）使用EMD作为Loss的原因

### 模型分层

如图2所示，所提出的网络包括三个组件：逐个标签特征地块学习模块，一个关注区域提取模块和一个标签关系推断模块。令L为对象标签的数量，l为第l个标签。标签方式的特征包学习模块被设计为针对每个标签l提取具有K个通道的高级特征图X1，称为特征包（更多细节参见II-B节）。注意区域提取模块用于对每个Xland中的可区分区域进行定位，以生成一个注意特征块A1，该块应该包含与标签l有关的最相关的语义。最后，由标签关系推断模块推论奥兰所有其他基于标签的关注特征地块之间的关系，以预测对象l的存在。其余部分介绍了建议的网络的详细信息。





#### \subsection{Inter-Class Feature Interaction Self-Adjustment}

类间特征交互自调节是
我们认为特征之间的交互行为存在“竞争”与“协作”两种形式，

我们提出“影响”的概念，以表达特征之间的交互作用

至此我们已经拥有了对应于每个类别的特征表示，接下来我们将会构建一种特征交互机制，令模型具备能力 根据类间相关性去自动调节特征内容以体现类别之间的相互作用。
$$
\{{F}_k^{spec}\}_{k=1}^{C}
$$




遵从键值形式的注意力机制的
$$
{Interact}({Q,K,V})_i = \frac{\sum_{j=1}^{C}{Eff}({q_i,k_j}) \cdot v_j}{\sum_{j=1}^{C}{Eff}({q_i,k_j})}\\
$$
where$q_i$, $k_i$ and $v_i$ are elements of matrix $\bm Q$, $\bm K$, and $\bm V$.


$$
{Eff}(\mathit{q_i,k_j}) = {\psi({q_i})}^{\top}{\phi({k_j})} + 1
$$

$$
\psi(\mathit{x}) = {tanh}(\mathit{x}),~~
	\phi(\mathit{x}) = {sigmoid}(\mathit{x})\\
$$

Inspired by the recent Transformer model [23], we implant
the multi-head mechanism adapted to our proposed feature interaction module. 













值得注意的是，我们的目的是希望利用类别间的关系来对标签分布的预测情况进行修正，即使其在表层形式上与自我注意力机制契合，行为目的却而并非是进行局部信息的聚焦或长期依赖的建模。

It is worth noting that our aim is to exploit the relationship between categories to revise the predicted situation of the label distribution, even if it is superficially formally compatible with the self-attention mechanism, with the behavioral aim but not the modeling of local information focus or long-term dependence.

顺便一提，我们所提出的方法本可以通过层叠多组线性前馈网络和自调节模块得到更高阶的特征交互关系，但考虑到数据量的因素，我们在本文中仅使用了一层。

As a side note, our proposed method could have yielded higher order feature interactions by cascading multiple sets of linear feedforward networks and self-tuning modules, but given the amount of data, we have used only one layer in this paper.

#### \subsection{Semantic-Enhanced Feature Representation}

$$
\mathbf{M}_k = {Softmax(g(\mathbf{E}\cdot L_k))}\\
{F}_k^{spec} =\mathbf{M}_k \cdot {{F}^{proj}_k}\\
[x_i,y_i].
$$

#### \subsection{Distribution-Oriented Model Learning}

$$
\hat{y} = \sigma(w^{\top}[F_{1}^{Ad},F_{2}^{Ad}...,F_{H}^{Ad}]+b) \\
[\cdot]
$$
其中$[\cdot]$表示级联操作, $w$表示投影权重, b表示偏置。

在模型最后输出预测结果时，常用的softmax layer并没有被采用以处理logit的值。这是因为我们认为Softmax对一个向量进行概率意义上的归一化的同时，会对于每个位置上的数值大小带来强烈的副作用。它会使数值间的大小差异更加明显，这对连续值预测的分布学习是不利的。因此$\sigma(x) = 1/(1+e^{-x})$被采用使得每个类别位置的描述值被限定在0到1区间范围内，这为模型输出的分布情况带来了更多的可能性。

作为常用的多变量回归任务损失函数，Kullback-Leibler (KL) divergence [28]也并没有被我们选择去对分布预测的差异进行惩罚。

We choose the Kullback-Leibler (KL) divergence [28] as the loss function to penalize
the deviation of the predicted distribution ˆ y from the ground-truth distribution y

原因之一在于其计算公式的非对称性使得该散度并不能成为严格的距离度量。此外，它的计算并未考虑概率取值的位置差异，可能会导致不同的分布预测相对同一个真实分布的差异是相同的。综上考虑，取而代之的是基于EMD[引用]的损失函数。这个更适用于度量分布之间距离的损失函数可被计算如下：

由于其计算方式与类别顺序相关，我们按照数据集中每类的得票数统计值的升序重新排列类标号，以避免模型忽略那些较少出现的类别



如图\ref{fig:density}中(a)所示。

为了更好地描述概率分布的取值情况，我们还设置了辅助的损失函数以拟合概率分布的取值密集程度。

这是出于我们对于数据的观察，概率分布的取值规律存在两种形式，即High-density以及Low-density。如图\ref{fig:density}中(b)和(c)所示，概率可能会集中分布在少数类别或者较为均匀地分散在多个类别，这些模式可以被定量描述以及被拟合。

As shown in (b) and (c) in Fig. \ref{fig:density}, probabilities may be concentrated in a few categories or evenly spread across many categories, and these patterns can be described quantitatively and fitted.

我们认为信息熵具备这种描述density的能力，它的计算公式如下：
$$
Dens(d) = -\sum_{k=1}^{C}d^{(k)}log(d^{(k)})
$$
特别地，当概率集中至单一的类别时，此式将取值为$0$，而完全均匀分布在每个类别时，此式将取值为$lnC$

这之后我们可以采用L1范数的形式计算真实概率分布与预测概率分布之间的差异作为辅助的损失函数：
$$
L_{regr}(d,~\hat{d}) = ||Dens(d) - Dens(\hat{d})||_{1}
$$
为两个损失函数部件添加权重因子，最终总的损失函数可以被定义如下





Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and
Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature
Interactions for Recommender Systems. In Proceedings of the 24th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining. ACM, 1754–
1763.

Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. 2016.
Deep crossing: Web-scale modeling without manually crafted combinatorial
features. In Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining. ACM, 255–262.





## 实验

![image-20201116102220883](C:\Users\15244\AppData\Roaming\Typora\typora-user-images\image-20201116102220883.png)

a=0.85, d=256, h=8

实验二：类特征维度=嵌入维度

| Factor_a | Dim  | Head | ACC     | COS     | KL      | Epoch |
| -------- | ---- | ---- | ------- | ------- | ------- | ----- |
|          | 64   |      | 626 635 | 801 803 | 529 533 |       |
|          | 128  |      | 629 634 | 797 802 | 549 574 |       |
|          | 256  |      | 713 709 | 851 854 | 426 442 |       |
|          | 512  |      | 624 630 | 798 803 | 569 549 |       |
|          | 1024 |      | 641     | 805     | 562     |       |

实验一：损失函数权重因子

| Factor_a | Dim  | Head | ACC     | COS     | KL      | Descri.                |
| -------- | ---- | ---- | ------- | ------- | ------- | ---------------------- |
| 0.7      |      |      | 623 623 | 783 779 | 660 663 | NoEb                   |
| 0.75     |      |      | 616 629 | 776 796 | 646 661 | NoEb                   |
| 0.8      |      |      | 618 629 | 756 794 | 613 694 | NoEb                   |
| 0.85     |      |      | 621 640 | 784 804 | 584 579 | NoEb                   |
| 0.9      |      |      | 638     | 803     | 617     | NoEb                   |
| 0.95     |      |      | 630 641 | 799 804 | 532 567 | NoEb                   |
| 1        |      |      | 678     | 802     | 481     | HasEb                  |
| 0.95     |      |      | 638 640 | 804 805 | 477 554 | HasEb Server           |
| 0.9      |      |      | 650 647 | 797 804 | 506 519 | HasEb Server           |
| 0.85     |      |      | 715 714 | 854 852 | 422 509 | HasEb Local            |
| 0.8      |      |      | 659     | 771     | 483     | HasEb Server wait...17 |
| 0.75     |      |      | 6236    | 706     | 523     | HasEb Server wait...15 |

实验三：头的个数

| Factor_a | Dim     | Head  | ACC     | COS     | KL      | Epoch |
| -------- | ------- | ----- | ------- | ------- | ------- | ----- |
|          |         | 1     | 629 634 | 800 807 | 533 534 | HasEb |
|          |         | 2     | 6299    | 801 828 | 528     | NoEb  |
|          |         | 4     | 629 641 | 802 834 | 508 546 | NoEb  |
| **0.95** | **256** | **8** | **683** | **845** | **468** | NoEb  |
|          |         | 16    | 634 636 | 801 801 | 541 573 | HasEb |







权重因子() 四点 x 头数(1,4,8,16)四点 = **16个实验**点*3指标 = 48点





特别地，头数为0表示使用基础的CNN模型，权重因子a为1时表示只有EMD based损失函数参与了优化。

随着超参数a的减小，衡量分布密度的回归损失函数占比增加，随之而来的是参与主要优化任务的分布损失函数占比降低。a取值在0.8以下的实验表现明显不好，因此我们未将这些结果列出。

超参数num_heads的增加会带来一定的性能提升，但

结论是，当头数=8，权重因子=0.95时模型在验证集上的情感分布预测效果达到最优，我们将使用这一组取值作为在测试集上进行对比试验的设定。

**JIEC里面啥都有**

Discrete Binary Coding based Label Distribution Learning

![image-20201025145159066](C:\Users\15244\AppData\Roaming\Typora\typora-user-images\image-20201025145159066.png)

EmoGCN

![image-20201123000637315](C:\Users\15244\AppData\Roaming\Typora\typora-user-images\image-20201123000637315.png)

从结果可以看出：（1）EFIA在所有数据集的大多数度量中均表现出优势，这证明了EFIA通过利用特征交互模块和面向分布的多任务学习在图像情感分布学习中的有效性； （2）即使我们提出的方法主要目标是预测情绪分布，EFIA也可以实现最佳的分类性能。 （3）对于Canber和Cosine测度，EFIA的表现比EGCN差，因为我们提出的方法更倾向优化概率分布的特性，EGCN更倾向于优化欧几里得空间的距离，这有利于Canber和Cosine的改善。







## 结论

在本文中，针对情感分布学习中的类间的相关性话题，我们提出带类间特征交互自调整模块进行的标签分布学习框架，从情感特征空间考虑类别特征表示间的相互影响以捕获不完全依赖静态知识的灵活的关系。同时，模型输出与损失函数的特殊设计 增强了模型对于分布学习任务的专门性。实验结果表明了我们提出的方法具备优势与合理性。

## 摘要

近年，视觉情感识别的视角逐渐从分类问题转向为标签分布问题。

大多数方法从情感语义的视角，通过先验知识对预测值进行限制或加权以利用类别关联性。

在本文中，我们提出了一种基于逐情感特征交互分析的情感分布学习框架(FISA)，以应对那些"复杂图像蕴含的整体情感分布不严格遵循普遍性心理学规律"的情况。从情感特征视角促进特定类别特征相互影响，来学习细致具体的类间关系。它同时考虑了情感语义与视觉表象，能够利用标签蕴含的静态关联与多变的图像特征的动态关联。

另外我们重构了模型部件以得到面向标签分布学习的专业模型。

在情感识别数据及上的大量实验结果证明了FISA的优势与合理性



## 参考

![image-20201024012124308](C:\Users\15244\AppData\Roaming\Typora\typora-user-images\image-20201024012124308.png)

美学分布数据集：Content-based photo quality assessment文章中已经把数据集放在百度网盘上了https://pan.baidu.com/s/1MBFuNpen6ushJfCp5Mx9wg，TMM上的Distribution-oriented Aesthetics Assessment with Semantic-Aware Hybrid Network可以比拼一下



#### 以后来源

* 本工作可以深入做，怎么真的生成类别关注图，而不使用标签嵌入？

* 特征层次相关性：

各层特征的Transformer



* **Caption的对象属性，**当然可以加入情感分析！



标签相关性：

Reconstruction Regularized Deep Metric Learning__for Multi-label Image Classification.pdf

Cross-Modality Attention with Semantic Graph Embedding for Multi-Label Classification

自监督学习：https://github.com/lightly-ai/lightly

@INPROCEEDINGS{2010micblog,
	author={G. {Li} and S. C. H. {Hoi} and K. {Chang} and R. {Jain}},
	booktitle={2010 IEEE International Conference on Data Mining}, 
	title={Micro-blogging Sentiment Detection by Collaborative Online Learning}, 
	year={2010},
	volume={},
	number={},
	pages={893-898},
	doi={10.1109/ICDM.2010.139}}

@INPROCEEDINGS{2020twitter,
	author={L. {Singh} and P. {Gupta} and R. {Katarya} and P. {Jayvant}},
	booktitle={2020 Fourth International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud) (I-SMAC)}, 
	title={Twitter data in Emotional Analysis - A study}, 
	year={2020},
	volume={},
	number={},
	pages={1301-1305},
	doi={10.1109/I-SMAC49090.2020.9243326}}



@article{kullback1951information,
	title={On Information and Sufficiency/The Annals of Mathematical Statistics.-Vol. 22.-N. 1},
	author={Kullback, S and Leibler, RA},
	journal={Institue of Mathematical Statistics},
	year={1951}
}





@INPROCEEDINGS{2018Fan,
	author={Y. {Fan} and H. {Yang} and Z. {Li} and S. {Liu}},
	booktitle={2018 11th International Congress on Image and Signal Processing, BioMedical Engineering and Informatics (CISP-BMEI)}, 
	title={Predicting Image Emotion Distribution by Emotional Region}, 
	year={2018},
	volume={},
	number={},
	pages={1-9},
	doi={10.1109/CISP-BMEI.2018.8633190}}

@ARTICLE{2019Fan,
	author={Y. {Fan} and H. {Yang} and Z. {Li} and S. {Liu}},
	journal={IEEE Access}, 
	title={Predicting Image Emotion Distribution by Learning Labels’ Correlation}, 
	year={2019},
	volume={7},
	number={},
	pages={129997-130007},
	doi={10.1109/ACCESS.2019.2939681}}



@inproceedings{bakshi2016opinion,
	title={Opinion mining and sentiment analysis},
	author={Bakshi, Rushlene Kaur and Kaur, Navneet and Kaur, Ravneet and Kaur, Gurpreet},
	booktitle={2016 3rd International Conference on Computing for Sustainable Global Development (INDIACom)},
	pages={452--455},
	year={2016},
	organization={IEEE}
}



另外，图1还是画得不行，没说明任何问题，而且居然也没在introduction中引用了，图肯定要保留，但要修改，修改成什么样子，我已经说过很多遍了，你自己考虑。