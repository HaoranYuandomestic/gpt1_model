# Abstract
本文提出了一种解决自然语言理解的模型，它首先经过一个生成式预训练模型 (GPT) ，然后经过鉴别式微调 (DF) 模型，本模型与以前的方法不同的是，本模型在微调阶段充分利用了已知任务的输入的变换以达到有效的转化并尽可能少的改变模型架构。

# Introduction
目前使用无监督学习解决 NLP 问题变成了一个非常有意义的课题，但是通过没有标签的文本充分利用 word-level 信息存在两个比较大的问题：
1. 目前还不知道什么样的优化方法对学习文字标识最有效
2. 目前对什么是将文字特征传输给目标任务的方法并不存在共识。
这些导致了半监督方法学习 NLP 变得非常困难。本篇论文使用了一种半监督的方法进行语言理解任务的处理，第一步利用一个语言模型对无标签数据去学习神经网络的初始参数，而第二步将这些参数投入目标任务，并利用监督学习方式进行参数学习，这里我们假设我们已经有了一系列的对无标注文本和数个数据集途径。

# Related Work
## Semi-supervised learning for NLP
## Unsupervised pre-training
## Auxiliary training objectives

# Framework
## Unsupervised pre-training
针对 `tokens` $\mathcal{U}=\{u_1,\cdots,u_n\}$ ，我们使用标准语言模型来最大化下面的似然函数：
$$L_q(\mathcal{U})=\sum\limits_i \log P(u_i|u_{i-k},\cdots,u_{i-1};\Theta)$$
其中 $k$ 就是 Markov 步数， $\Theta$ 就是神经网络的参数，这里我们可以使用 SGD 优化。本文使用 Transformer decoder 作为语言模型，其本质是 transformer 的一个变种，它首先使用一个 multi-headed self-attention 层进行计算，并紧接一个全连接网络生成输出目标 token 的概率分布：
$$h_0=UW_e+W_p$$
$$h_l=transformer_block(h_{l-1})\ \forall i\in[1,n]$$
$$P(u)=softmax(h_nW_e^T)$$
其中 $U=(u_{-k},\cdots,u_{-1})$ ， $n$ 是层数， $W_e$ 是压缩矩阵， $W_p$ 是位置压缩矩阵。

## Supervised fine-tuning
下面进行监督学习，对有标注的数据集 $\mathcal{C}$ ，其中每个数据都有输入序列 $x^1,\cdots,x^m$ 和标注 $y$ ，每当输入一个序列 $x$ 我们都可以通过前面的预训练模型得到序列 $h_l$ 我们将它带入下面的式子计算：
$$P(y|x^1,\cdots,x^m)=softmax(h_l^mW_y)$$
进而可以最大化下面的似然函数
$$L_2(\mathcal{C})=\sum\limits_{(x,y)}\log P(y|x^1,\cdots,x^m)$$
这样我们可以实现提高监督模型的泛化能力，并且加速收敛，此时我们对损失函数做进一步优化：
$$L_3(\mathcal{C})=L_2(\mathcal{C})+\lambda*L_1(\mathcal{C})$$

## Task-specific input transformations
针对不同的任务可能有不同特征的输入，我们这里对输入的转化做出了一些规范，但是由于与机器翻译并不相关，这里我们不再作过多赘述。

# Experiments
## Setup
### Unsupervised pre-training
我们使用 BooksCorpus 作为我们的训练数据集，另外可以作为替代的数据集为 1B Word Benchmark.

### Model specifications
我们主要来介绍一下系数

