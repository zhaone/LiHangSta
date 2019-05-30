EM&GMM
----------------------------------------------------
# EM算法
EM算法是一种**迭代算法**，用于**含有隐变量**（hidden variable）的概率模型参数的**极大似然估计**，或**极大后验概率估计**。EM算法的每次迭代由两步组成：**E步，求期望（expectation）**；**M步，求极大（maximization）**。所以这一算法称为**期望极大算法（expectation maximization algorithm）**，简称EM算法。
> actually, 李这一章是在为后面的隐马尔科夫和条件随机场的学习打算法基础
## 我是栗子
什么叫做含有因变量的概率模型呢，来个栗子，李书里面给出的是三硬币模型。
> 仨硬币A,B,C，正面出现的概率分别为a,b,c。单次实验是，首先掷A硬币，若A为正，掷B硬币，若A为反，掷C硬币，实验结果为第二个硬币的正反，正记为1，反记为0。现在给一个实验观测序列$Y=\{1,1,0,1,0,0,1,0,1,1\}$，采用极大似然估计求a,b,c。

则按照普通的想法，这不就是个极大似然估计吗。我们为第$i$次实验设置一个变量$z_i$，当A为正时，$z_i=1$，否则$z_i=0$。
那么
$$
P(y|z,\theta)=\prod_j^N ab^{z_i}(1-b)^{1-z_i}+(1-a)c^{z_i}(1-c)^{1-z_i}
$$
那么对数似然函数要使$P(y|\theta)$最大，
$$
\log \sum_Z P(y|z,\theta)=\log(\sum_Z\prod_j^N ab^{z_i}(1-b)^{1-z_i}+(1-a)c^{z_i}(1-c)^{1-z_i})
$$
对这个式子对$a,b,c,z_i$求偏导，是没有解析解的...不知道为啥没有，所以无法直接通过上式求解。



## EM算法的导出

那么问什么最大化上述式子能得到问题的解呢。下面通过近似求解观测数据的对数似然函数的极大化问题来导出EM算法，由此可以清楚地看出EM算法的作用。

假设我们有一个含有隐藏变量$Z$的对数似然函数$L(Y|\theta)$，其中$Y$为观测值，$\theta$为求解的参数。**最大化对数似然函数即最大化观测值$Y$对参数$\theta$的对数似然函数**，即下式
$$
L(\theta)=\log{P(Y|\theta)}=\log{\sum_ZP(Y,Z|\theta)}=\log{\sum_ZP(Y|Z,\theta)P(Z|\theta)}
$$
注意到这一极大化的主要困难是式6中有未观测数据并有包含和（或积分）的对数。。。没考证过，您说有就有吧。

那一个新办法，就是通过迭代的方法逐步使得$L(\theta)$增大。假设第$i$轮迭代得到的参数是$\theta_i$，我们想达到的目的是$L(\theta_{i+1})-L(\theta_i)>0$，现在假设我们已经有了第$i$轮的参数$\theta_i$，求使得$L(\theta)-L(\theta_i)$最大的$\theta$，有
$$
L(\theta)-L\left(\theta^{(i)}\right)=\log \left(\sum_{Z} P(Y | Z, \theta) P(Z | \theta)\right)-\log P\left(Y | \theta^{(i)}\right)
$$
由Jensen不等式（其实我不知道Jensen不等式为撒是对的）得
$$
\begin{aligned} L(\theta)-L\left(\theta^{(i)}\right) &=\log \left(\sum_{z} P\left(Y | Z, \theta^{(i)}\right) \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Y | Z, \theta^{(i)}\right)}\right)-\log P\left(Y | \theta^{(i)}\right) \\ & \geqslant \sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right)}-\log P\left(Y | \theta^{(i)}\right) \\ &=\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)} \end{aligned}
$$
由于$L(\theta^i)$是定值，使得$L(\theta)$最大化的$\theta^{i+1}$等价于下式
$$
\begin{aligned}\theta^{(i+1)} &= \arg \max_\theta\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)}\\&=\arg \max_\theta \sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log P(Y, Z | \theta)\\&=\arg \max_\theta Q(\theta,\theta^{(i)})\end{aligned}
$$
（这一块我觉得李书讲的有点复杂，不过他最后那个对$\Q(\theta,\theta^{(i)})$图比较形象，可以看下，图中的$B(\theta,\theta^{(i)})=L(\theta^{(i)}+Q(\theta,\theta^{(i)})$此处不写了。

第二节李还证明了EM算法的收敛性，很容易看懂，不再赘述了。

# 高斯混合模型

[b站讲解](https://www.bilibili.com/video/av35183585/?p=3)这里的推导我是按照b站一个up主的思路推导的，实在是因为**李书里面的推导我推不出来。。。**高斯混合模型是指具有如下形式的概率分布模型：
$$
P(x | \theta)=\sum_{k=1}^{K} \alpha_{k} \phi\left(x | \theta_{k}\right)
$$
就是多个高斯分布的加权和，其中$\alpha_k$为第$k$个高斯分布的权，$\sum_{k=1}^K{\alpha_k}=1$，$\theta_k=\{\mu_k,\delta_k\}$，为第$k$个高斯分布的参数。假如说有一组观测值$x_1,x_2,...,x_n$，现在我们用该观测值估算所有的参数。

## E-step

首先我们要确定观测变量和隐变量，对于观测值$x_j$，我们可以这么想，它是由第$k$个高斯模型产生的，为$x_j$设置一个隐变量$z_j$，则$z_j=1\ or\ 2,...,or\ K$，所以完全变量为$x_j,z_j$。

所以我们有
$$
P(x_j,z_j|\theta)=P(x_j|z_j,\theta)*P(z_j|\theta)=\alpha_{z_j}\phi(x_j|\theta_{z_j})
$$
同时有
$$
P(z_j|x_j,\theta)=\frac{P(z_j,x_j|\theta^{(i)})}{P(x_j|\theta^{(i)})}=\frac{\alpha_{z_j}\phi(x_j|\theta_{z_j})}{\sum_{k=1}^{K} \alpha_{k} \phi\left(x_j | \theta_{k}\right)}
$$


6,7两个式子应该是比较好理解的，解释下吧。对于式6，$x_j,z_j$同时发生，即首先$x_j$由正态分布$\phi(x|\theta_{z_j})$产生，这个概率为$\alpha_{z_j}$，然后正态分布$\phi(x|\theta_{z_j})$产生$x_j$的概率为$\phi(x_j|\theta_{z_j})$，两者应该是独立的所以两相乘者$P(x_j,z_j|\theta)=\alpha_{z_j}\phi(x_j|\theta_{z_j})$；对于式7，这个貌似没甚好说的，直接条件概率公式。

写出$Q$函数
$$
\begin{aligned}
Q(\theta,\theta^{(i)})&=\sum_Z \log P(x,z|\theta) P(z|x,\theta^{(i)})\\
&=\sum_Z \log (\prod_j^N P(x_j,z_j|\theta)) P(z|x,\theta^{(i)})\\
&=\sum_Z [\sum_{j=1}^N\log(P(x_j,z_j|\theta))P(z|x,\theta^{(i)})]
\end{aligned}
$$
解释下$Z,z,z_j$的意思，要不会及其混乱。

- $z_j$即$x_j$对应的隐变量
- $z$指整个的隐变量序列，即$z_1,z_2,...,z_n$
- $Z$指所有可能的$z$组成的集合，可以理解为一个K*N的矩阵，每行表示一个$z$，第$i$列表示$z_i$所有可能的取值，公式里的$\sum_Z$是指每次取一行，得到一个$z$

我们继续化简上式，
$$
\begin{aligned}
Q(\theta,\theta^{(i)})&=\sum_Z [\sum_{j=1}^N\log(P(x_j,z_j|\theta))P(z|x,\theta^{(i)})]\\
&=\sum_{j=1}^N [\sum_Z \log P(x_j,z_j|\theta) P(z|x,\theta^{(i)})]
\end{aligned}
$$
我们单提出$j=1$的情况来看
$$
\begin{aligned}
&\sum_Z \log P(x_1,z_1|\theta) P(z|y,\theta^{(i)})\\
=&\sum_Z \log P(x_1,z_1|\theta) \prod_{j=1}^N P(z_j|x_j,\theta^{(i)})\\
=&\sum_Z \log P(x_1,z_1|\theta)P(z_1|x_1,\theta^{(i)}) \prod_{j=2}^N P(z_j|x_j,\theta^{(i)})\\
\end{aligned}
$$
其中$ \log P(x_1,z_1|\theta^{(i)})P(z_1|x_1,\theta^{(i)})$之和$Z_1$有关系，$\prod_{j=2}^N P(z_j|x_j,\theta^{(i)})$和$Z_{2,3,...,N}$有关系，所以上式子可以改写为
$$
\begin{aligned}
&\sum_{Z_1} \log P(x_1,z_1|\theta)P(z_1|x_1,\theta^{(i)}) \sum_{Z_2,...,Z_N}\prod_{j=2}^N P(z_j|x_j,\theta^{(i)})\\
=&\sum_{Z_1} \log P(x_1,z_1|\theta)P(z_1|x_1,\theta^{(i)}) 
\end{aligned}
$$
这是因为$\sum_{Z_2,...,Z_N}\prod_{j=2}^N P(z_j|x_j,\theta^{(i)})$相当于取了所有的概率分布情况，其和肯定是1。由此我们得到了单提出$j=1$的结果，那么所有的和即为
$$
\sum_{j=1}^N \sum_{Z_j} \log P(x_j,z_j|\theta)P(z_j|x_j,\theta^{(i)})
$$
将公式6,7代入得到
$$
\begin{aligned}
Q(\theta,\theta^{(i)})&=\sum_{j=1}^N \sum_{Z_j} \log P(x_j,z_j|\theta)P(z_j|x_j,\theta^{(i)})\\
&=\sum_{j=1}^N \sum_{Z_j} \log(\alpha_{z_j}\phi(x_j|\theta_{z_j}))\frac{\alpha_{z_j}^{(i)}\phi(x_j|\theta_{z_j}^{(i)})}{\sum_{k=1}^{K} \alpha_{k}^{(i)} \phi\left(x_j | \theta_{k}^{(i)}\right)}\\
&=\sum_{j=1}^N \sum_{k=1}^K \log(\alpha_{k}\phi(x_j|\theta_{k}))\frac{\alpha_{k}^{(i)}\phi(x_j|\theta_{k}^{(i)})}{\sum_{k=1}^{K} \alpha_{k}^{(i)} \phi\left(x_j | \theta_{k}^{(i)}\right)}
\end{aligned}
$$


## M-step

上面的式子的后面一坨太麻烦了， 我们还是把它写回来（反正大家知道他是啥就行），同时为了求解方便，我们将上式的求和符号翻一翻（这显然是无所谓的）
$$
\begin{aligned}
Q(\theta,\theta^{(i)})&=\sum_{k=1}^K  \sum_{j=1}^N \log(\alpha_{k}\phi(x_j|\theta_{k}))P(z_j=k|x_j,\theta^{(i)})\\
&=\sum_{k=1}^K  \sum_{j=1}^N [\log(\alpha_{k})+\log \phi(x_j|\theta_{k}))]P(z_j=k|x_j,\theta^{(i)})\\
\end{aligned}
$$
好那么我们开始求解了。首先我们求$\alpha_k$
$$
\alpha_k=\arg \max_{\alpha_k}   \sum_{j=1}^N [\log(\alpha_{k})+\log \phi(x_j|\theta_{k}))]P(z_j=k|x_j,\theta^{(i)})\\
$$
显然$\log \phi(x_j|\theta_{k})$这一坨和$\alpha_k$没有什么卵关系，所以可以写为
$$
\alpha_k=\arg \max_{\alpha_k}   \sum_{j=1}^N \log\alpha_{k} P(z_j=k|x_j,\theta^{(i)})\\
s.t. \sum_{k=1}^K\alpha_k-1=0
$$
这是一个带约束的最大化问题，可以用拉格朗日乘数法，设$\sum_{k=1}^K\alpha_k-1=0$的拉格朗日乘子为$\lambda$，则我们对式子
$$
\sum_{j=1}^N \log(\alpha_{k})P(z_j=k|x_j,\theta^{(i)})+\lambda(\sum_{k=1}^K\alpha_k-1)
$$
求$\alpha_k$的倒数，有
$$
\sum_{j=1}^N \frac{P(z_j=k|x_j,\theta^{(i)})}{\alpha_k}+\lambda=0\\
\sum_{j=1}^N P(z_j=k|x_j,\theta^{(i)})+\alpha_k\lambda=0
$$
求所有和得到
$$
\sum_{k=1}^K \sum_{j=1}^N P(z_j=k|x_j,\theta^{(i)})+\alpha_k\lambda=0\\
\sum_{k=1}^K \sum_{j=1}^N P(z_j=k|x_j,\theta^{(i)})+\sum_{k=1}^K\alpha_k\lambda=0\\
\lambda=-\sum_{k=1}^K \sum_{j=1}^N P(z_j=k|x_j,\theta^{(i)})\\
\lambda=-\sum_{j=1}^N \sum_{k=1}^K P(z_j=k|x_j,\theta^{(i)})\\
\lambda=-\sum_{j=1}^N 1=-N\\
$$
将$\lambda$代入式12，得到
$$
\alpha_k=\frac{\sum_{j=1}^N P(z_j=k|x_j,\theta^{(i)})}{N}
$$
其中$P(z_j=k|x_j,\theta^{(i)})=\frac{\alpha_{k}^{(i)}\phi(x_j|\theta_{k}^{(i)})}{\sum_{k=1}^{K} \alpha_{k}^{(i)} \phi\left(x_j | \theta_{k}^{(i)}\right)}$。

按照同样的方法可以求出$\mu_k,\delta_k$。

# EM算法的推广

没看。。。