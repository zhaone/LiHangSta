隐马尔科夫模型
-----------------------------

# 背景知识

二话不说先偷图，下图给出了HMM在概率图模型中的地位。

![HMM](E:\LiHangSta\HMM\背景知识.JPG)

从图中可以看出，HMM是一种概率图模型。概率图模型根据是否有向分为贝叶斯网络和马尔科夫随机场。如果概率图加上时间序列，就变成了动态模型（这个时间序列是指概率图模型的多个状态之间不是独立同分布的）。动态模型如果状态分布是离散的，那么就是隐马尔科夫模型，如果是连续的，根据分布是线性还是非线性分为Kalman Filter和Particle Filter。例如GMM（不是概率图模型），其观测值之间没有关系，独立同分布；但是HMM状态之间有联系，就是动态模型。

EMM在NLP方向应用比较广，适合用于解决**序列标注问题**。

# HMM基本概念

偷图大法好，下图是HMM模型的示意图。

![HMM模型](E:\LiHangSta\HMM\HMM模型.JPG)

隐马尔可夫模型是关于时序的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。

- 一个模型
  - 隐藏的马尔可夫链随机生成的状态的序列，称为状态序列（state sequence）$i_1,i_2,...,i_t,...,i_T$
  - 每个状态生成一个观测，而由此产生的观测的随机序列，称为观测序列（observation sequence）$o_1,o_2,...,o_t,...,o_T$
  - 状态序列的取值集合记作$Q=\left\{q_{1}, q_{2}, \cdots, q_{N}\right\}$
  - 观测序列的取值集合记作$V=\left\{v_{1}, v_{2}, \cdots, v_{M}\right\}$
  - $T$为状态序列（观测序列）长度
- 两个基本假设
  - 齐次马尔可夫假设，即$i_t+1$只和$i_t$有关，公式表达为$P\left(i_{t} | i_{t-1}, o_{t-1}, \cdots, i, o_{1}\right)=P\left(i_{t} | i_{t-1}\right), \quad t=1,2, \cdots, T$
  - 观测独立性假设，即$o_t$只和$i_t$有关，公式表达为$P(o_t|i_T,O_t,i_{T-1},o_{T-1},....,i_t,...,i_1,o_1)=P(o_t|i_t)$
- 三个参数，记作$\lambda=\{A,B,\pi\}$
  - A为状态转移矩阵，其大小为$N*N$，$a_{i j}=P\left(i_{t+1}=q_{j} | i_{t}=q_{i}\right), \quad i=1,2, \cdots, N ; j=1,2, \cdots, N$
  - B为观测概率矩阵（发射矩阵），其大小为$N*M$，其中$b_{jk}=P\left(o_{t}=v_{k} | i_{t}=q_{j}\right), \quad k=1,2, \cdots, M ; j=1,2, \cdots, N$
  - $\pi$为初始状态概率，公式$\pi_i=P(i_1=q_i)$

- 三个问题
  - 计算问题。给定参数$\lambda$，观测序列$O=\{o_1,o_2,...,o_T\}$，求$P(O|\lambda)$
  - 学习问题。给定观测序列$O=\{o_1,o_2,...,o_T\}$，估计模型参数$\lambda$
  - 预测问题。给定参数$\lambda$，状态序列$I=\{i_1,i_2,...,i_T\}$，求最优可能的观测序列$O$

# 计算问题

>计算问题。给定参数$\lambda$，观测序列$O=\{o_1,o_2,...,o_T\}$，求$P(O|\lambda)$

$O$是由$I$产生的，所以$I$其实是隐变量，完整变量可以记为$O,I$，$P(O|\lambda)=P(O|I,\lambda)*P(I|\lambda)$。按照一般的方法，
$$
\begin{aligned}
P(O|\lambda)&=\sum_IP(O|I,\lambda)*P(I|\lambda)\\
&=\sum_{I_1,I_2,...,I_T}P(O|i_1,i_2,...,i_T,\lambda)*P(i_1,i_2,...,i_T|\lambda)\\
&=\sum_{j_1=1}^N\sum_{j_2=1}^N,...,\sum_{j_T=1}^NP(O|i_1=q_{j_1},i_2=q_{j_2},...,i_T=q_{j_T},\lambda)*P(i_1=q_{j_1},i_2=q_{j_2},...,i_T=q_{j_T}|\lambda)\\
&=\sum_{j_1=1}^N\sum_{j_2=1}^N,...,\sum_{j_T=1}^N \pi_{j_1}b_{j_1 o_1} a_{j_1 j_2}b_{j_2 o_2},...,a_{j_{T-1} j_T}b_{j_T o_T}
\end{aligned}
$$
这复杂度为$O(TN^T)$，怎么感觉是TNT炸药，这复杂度为指数级别，谁也顶不住。所以要另想办法。

## 前向算法

我们记
$$
\alpha_t(i)=P(i_t=q_i,o_1,o_2,...,o_t|\lambda)
$$
所以
$$
\begin{aligned}
\alpha_{t+1}(i)&=P(i_{t+1}=q_i,o_1,o_2,...,o_t,o_{t+1}|\lambda)\\
&=\sum_{j=1}^NP(i_{t+1}=q_i,o_{t+1}|i_t=q_j,o_1,o_2,...,o_t,\lambda)P(i_t=q_j,o_1,o_2,...,o_t|\lambda)\\
&=\sum_{j=1}^NP(o_{t+1}|i_{t+1}=q_i,\lambda)P(i_{t+1}=q_i|i_t=q_j,o_1,o_2,...,o_t,\lambda)P(i_t=q_j,o_1,o_2,...,o_t|\lambda)\\
&=\sum_{j=1}^Nb_{io_{t+1}}a_{ji}\alpha_t(j)=b_{io_{t+1}}\sum_{j=1}^Na_{ji}\alpha_t(j)
\end{aligned}
$$

综合起来，有
$$
\alpha_t(i)=\left\{
\begin{array}{rcl}
&\pi_ib_{io_1} & t=1\\
&b_{io_{t}}\sum_{j=1}^Na_{ji}\alpha_{t-1}(j) & t>1
\end{array} 
\right.
$$
所以，我们最后只需要求
$$
P(o_1,o_2,...,o_T|\lambda)=\sum_j^NP(i_T=q_j,o_1,o_2,...,o_T|\lambda)=\sum_j^N\alpha_T(j)
$$
分析下式4的复杂度，观察式2，我们现在只关心相邻的两个时间$t-1,t$，求已知所有$\alpha_{t-1}(j),j=1,2,...,N$的情况下，求出所有$\alpha_{t}(i),i=1,2,...,N$的复杂度。显然根据$b_{io_{t}}\sum_{j=1}^Na_{ji}\alpha_{t-1}(j)$式可知，求一个$\alpha_t(i)$需要N+1次乘法，N-1加法，为2N次操作，所有求出所有$\alpha_{t}(i),i=1,2,...,N$需要$2N^2$次操作。又有时间$T$序列中有$T-1$个这种相邻层计算，另外还有$t=1$时的N次乘法和$t=T$时的N次加法，所以总的操作次数为$2N^2*(T-1)+2N$，复杂度为$N^2T$，比上文的$O(TN^T)$小了很多（当$N,T$较大的时候）。

## 后向算法

和前向算法思路差不多，我们记
$$
\beta_t(i)=P(o_{t+1},o_{t+2},...,o_T|i_t=q_i,\lambda)
$$
所以
$$
\begin{aligned}
\beta_{t-1}(i)&=P(o_{t},o_{t+1},o_{t+2},...,o_T|i_{t-1}=q_i,\lambda)\\
&=P(o_{t}|i_{t-1}=q_i,\lambda)P(o_{t+1},o_{t+2},...,o_T|i_{t-1}=q_i,\lambda)\\
&=\sum_j^NP(o_{t},i_t=q_j|i_{t-1}=q_i,\lambda)\beta_t(j)\\
&=\sum_j^NP(o_{t}|i_t=q_j,\lambda)P(i_t=q_j|i_{t-1}=q_i,\lambda)\beta_t(j)\\
&=\sum_j^Na_{ij}b_{jo_t}\beta_t(j)
\end{aligned}
$$
综上，所以
$$
\beta_t(i)=\left\{
\begin{array}{rcl}
&1 & t=T\\
&\sum_j^Na_{ij}b_{jo_{t+1}}\beta_{t+1}(j) & t>1
\end{array} 
\right.
$$
我们最后只需要求
$$
\sum_{i=1}^N\pi_ib_{io_1}\beta_1(i)
$$
具体复杂度就不分析了，也是$O(N^2T)$的。

## 其他

有时候还会用到其他概率的计算，例如下面的几个

### 时刻t处于状态$q_i$的概率

时刻t处于状态$q_i$的概率可以写成$P(i_t=q_i|O,\lambda)$。
$$
\begin{aligned}
\gamma_t(i)&=P(i_t=q_i|O,\lambda)\\
&=\frac{P(i_t=q_i,O|\lambda)}{P(O|\lambda)}\\
&=\frac{P(i_t=q_i,O|\lambda)}{\sum_jP(i_t=q_j,O|\lambda)}
\end{aligned}
$$
下面说明$P(i_t=q_i,O|\lambda)=\alpha_t(i)\beta_t(i)$，我们有
$$
\begin{aligned}
P(i_t=q_i,O|\lambda)&=P(i_t=q_i,o_1,o_2,...,o_T|\lambda)\\
&=...
\end{aligned}
$$

emmm说实话我没证明出来，但是书上说根据定义可知$P(i_t=q_i,O|\lambda)=\alpha_t(i)\beta_t(i)$，我TM就暂时信了吧，我的直觉和理性告诉我应该是这样。理论上应该是利用两个基本假设将上式化为$\alpha_t(i)\beta_t(i)$。

### 时刻$t$处于状态$q_i$且时刻$t+1$处于状态$q_j$的概率

记为$\xi_t(i,j)=P(i_t=q_i,i_{t+1}=q_j|O,\lambda)$，其实我自己也推不出来。。。这和上面拉那个问题是一样的，都是根据定义得到
$$
\xi_{t}(i, j)=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)}{P(O | \lambda)}=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)}{\sum_{i=1}^{N} P\left(i_{i}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)}
$$
感觉有点道理，且$P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)=\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)$，所以
$$
\xi_{t}(i, j)=\frac{\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}{\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}
$$

# 预测问题

> 给定状态序列$O$，求最大可能出现的观测序列$I$。个人觉得这是一个比较简单的问题，也比较好理解。

## 近似算法

前面我们已经算过时刻t处于状态$q_i$的概率$\gamma_t(i)=P(i_t=q_i|O,\lambda)$，那么近似算法确定$t$时刻状态$i_t$的方式为
$$
i_t=\arg \max_{q_i}\gamma_t(i)
$$
即$t$时刻最有可能的状态，简单粗暴，得出答案。

## 维比特算法（动态规划）

上述近似算法比较简单，但是存在某些问题。例如按照上述方法求出$i_{t-1}=q_m$和$i_t=q_n$，可能$a_{mn}=0$即根本不存在$q_m$到$q_n$的可能性，这就不符合实际了。

|       | $o_1$        | $o_2$        | ...  | $o_{T}$      |
| ----- | ------------ | ------------ | ---- | ------------ |
| $q_1$ | $\Phi_{1,1}$ | $\Phi_{1,2}$ |      | $\Phi_{1,T}$ |
| $q_2$ | $\Phi_{2,1}$ | $\Phi_{2,2}$ |      | $\Phi_{2,T}$ |
| ...   |              |              |      |              |
| $q_T$ | $\Phi_{T,1}$ | $\Phi_{T,2}$ |      | $\Phi_{T,T}$ |



维比特算法的思想是动态规划，其中$\Phi_{i,t}$表示在观测序列为$o_1,o_2,...,o_t$，出现概率最大的状态序列$i_1,i_2,...,i_t$（其中$i_t=q_i$)的概率，显然$\Phi_{i,t}=\max a_{ji}\Phi_{jt-1}$。将上表填完之后，再记$\Psi_T=\arg \max_i \Phi_{i,T}$，且$\Psi_t=\arg \max_j \Phi{jt}*a_{j\Psi_t+1}$。这种方法的原理就是动态规划，对于学过一点算法的都比较容易理解，其背后原理相信不用做过多介绍了。

# 学习算法

> 给定一组训练数据$\{(O_1,I_1),(O_2,I_2),...,(O_S,I_S)\}$，求隐马尔科夫模型参数$\lambda$。

## 监督学习方法

用极大似然估计，我们有三个参数需要估计

- $\pi_i$为$S$个样本中，初始状态为$q_i$的频率
- 记样本状态为$j$且观测为$v_k$的频数为$B_{jk}$，那么$b_{jk}=\frac{B_{jk}}{\sum_{k=1}^MB_{jk}}$
- 记样本中$t$时刻状态为$q_i$且$t+1$时刻状态为$q_j$的频数为$A_{ij}$，则$a_{ij}=\frac{A_{ij}}{\sum_{j=1}^NB_{ij}}$

这都没什么好说的，用频率估计参数。

## 无监督学习方法

大部分情况下，对状态序列的标注比较麻烦，耗费大量的人工。无监督学习方法，通过对观测序列$O_1,O_2,...,O_S$的学习得到所有参数。那么此时$O$对应的$I$就是隐变量，完全变量为$O,I$，此时可以用EM算法解决该问题（貌似还叫Baum-Welch算法，因为那个时候EM算法还没提出来）。

首先我们写出完全变量，显然在该问题中，完全变量为$O，I$，那么$Q$函数为
$$
Q(\theta,\theta^{(i)})=\sum_I\log[P(O,I|\theta)]P(I|\theta^{(i)})
$$
其中$\theta$就是我们隐马尔科夫模型中的参数$\lambda=\{a,b,\pi\}$。

显然我们有
$$
P(O,I|\theta)=\pi_{i_1}b_{i_1o_1}a_{i_1i_2}b_{i_2o_2}...a_{i_{t-1}i_T}b_{i_To_T}\\
P(I|O,\theta^{(i)})=\frac{P(O,I|\theta^{(i)})}{P(O|\theta^{(i)})}
$$
则
$$
\begin{aligned}
Q(\theta,\theta^{(i)})&=\sum_I\log[P(O,I|\theta)]P(I|O,\theta^{(i)})\\
&=\sum_I[\log(\pi_{i_1})+\log(b_{i_1o_1})+\log(a_{i_1,i_2})+\log(b_{i_2o_2})+...+\log(a_{i_{T-1},i_T})+\log(b_{i_To_T})]P(I|O,\theta^{(i)})\\
&=\sum_I\log(\pi_{i_1})P(I|O,\theta^{(i)})+\sum_I[\log(a_{i_1,i_2})+...,+\log(a_{i_{T-1},i_T})]P(I|O,\theta^{(i)})+\sum_I[\log(b_{i_1o_1})+,...,+\log(b_{i_To_T})]P(I|O,\theta^{(i)})
\end{aligned}
$$

显然$Q$函数根据参数被分成了三坨无关的东西，要最大化$Q$函数我们只需要分别最大化每一坨就行了。

### 求初始分布概率

先看第一坨
$$
\begin{aligned}
\sum_I\log(\pi_{i_1})P(I|O,\theta^{(i)})&=\sum_I\log(\pi_{i_1})\frac{P(O,I|\theta^{(i)})}{P(O|\theta^{(i)})}\\
&=\frac{1}{P(O|\theta^{(i)})}\sum_I\log(\pi_{i_1})P(O,I|\theta^{(i)})\\
&=\frac{1}{P(O|\theta^{(i)})}\sum_I\log(\pi_{i_1})\pi^{(i)}_{i_1}b_{i_1o_1}a^{(i)}_{i_1i_2}b_{i_2o_2}...a^{(i)}_{i_{T-1}i_T}b_{i_{T-1}o_T}\\
&=\frac{1}{P(O|\theta^{(i)})}\sum_{I_1}[\log(\pi_{i_1})\sum_{I_2,...,I_T}\pi^{(i)}_{i_1}b_{i_1o_1}a^{(i)}_{i_1i_2}b_{i_2o_2}...a^{(i)}_{i_{T-1}i_T}b_{i_{T-1}o_T}\\
&=\frac{1}{P(O|\theta^{(i)})}\sum_{j=1}^N[\log(\pi_{j})P(i_1=q_j,O|\theta^{(i)})]\\
&=\sum_{j=1}^N[\log(\pi_{j})\frac{P(i_1=q_j,O|\theta^{(i)})]}{P(O|\theta^{(i)})}\\
&=\sum_{j=1}^N[\log(\pi_{j})P(i_1=q_j|O,\theta^{(i)})]\\
s.t.\sum_{j=1}^N\pi_j&=1
\end{aligned}
$$
引入拉格朗日乘子$\mu$，则对下式对$\pi_j$求导
$$
\begin{aligned}
&\frac{\partial \sum_{j=1}^N[\log(\pi_{j})P(i_1=q_j|O,\theta^{(i)})]+\mu(\sum_{j=1}^N\pi_j-1)}{\partial \pi_j}\\
&=\frac{P(i_1=q_j|O,\theta^{(i)})}{\pi_j}+\mu\\
&=0\\
equal\ to:\ &P(i_1=q_j|\theta^{(i)})+\pi_j\mu=0
\end{aligned}
$$
则对所有$j$求和，得到
$$
\sum_{j=1}^NP(i_1=q_j|O,\theta^{(i)})+\pi_j\mu=0\\
\mu=-\sum_{j=1}^NP(i_1=q_j|O,\theta^{(i)})
$$
将$\mu$代入$\frac{P(i_1=q_j|\theta^{(i)})}{\pi_j}+\mu=0$得到
$$
\pi_j=\frac{P(i_1=q_j|O,\theta^{(i)})}{\sum_{j=1}^NP(i_1=q_j|O,\theta^{(i)})}=P(i_1=q_j|O,\theta^{(i)})=\gamma_1(j)
$$
$\gamma_1(j)$见上文的计算问题，即$t$时刻状态$q_j$的概率。

### 求状态转移概率和发射矩阵

讲真，，，这两个神奇的a，b我真的不知道怎么化简成正常的$\sum$形式，看李书上的化简方式，直觉上是对的。书上是这么化简的
$$
\sum_{I}\left(\sum_{t=1}^{T-1} \log a_{i, i_{i+1}}\right) P(O, I | \overline{\lambda})=\sum_{i=1}^{N} \sum_{j=1}^{N} \sum_{t=1}^{T-1} \log a_{i j} P\left(O, i_{t}=i, i_{t+1}=j | \overline{\lambda}\right)\\
\sum_{I}\left(\sum_{t=1}^{T} \log b_{i}\left(o_{t}\right)\right) P(O, I | \overline{\lambda})=\sum_{j=1}^{N} \sum_{t=1}^{T} \log b_{j}\left(o_{t}\right) P\left(O, i_{t}=j | \overline{\lambda}\right)
$$


讲真，有了化简之后得公式，拉格朗日乘数法是比较简单的，然而。。。上面的两个式子我自己得不出来。

# 问题

主要是**推导类型**的问题

- 时刻t处于状态$q_i$的概率：没有推导过程，**根据定义**得来，我没有推导出来
- 时刻$t$处于状态$q_i$且时刻$t+1$处于状态$q_j$的概率：没有推导过程，**根据定义**得来，我没有推导出来
- 状态转移矩阵和发射矩阵的**求导前的化简形式**我没有化简出来