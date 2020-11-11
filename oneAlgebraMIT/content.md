# 2. Multiple Qubits and Entanglement

## 2.1 Introduction

We have seen some interesting effects with single qubits and single qubit gates, but the true power
of quantum computing is realised through the interactions between qubits. In this section we will
introduce multiple qubit gates and explore the interesting behaviours of multi-qubit systems.

```latex
\begin{cases}
2x&-y&&=0\\-x&+2y&-z&=-1\\&-3y&+4z&=4
\end{cases}
```

```latex
x\begin{bmatrix}
2\\-1\end{bmatrix}
+y
\begin{bmatrix}-1
\\2\end{bmatrix}
=
\begin{bmatrix}0\\3
\end{bmatrix}
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

x = [-2, 2, -2, 2]
y = [-4, 4, 0.5, 2.5]

fig = plt.figure()
plt.axhline(y=0, c='black')
plt.axvline(x=0, c='black')

plt.plot(x[:2], y[:2], x[2:], y[2:])

plt.draw()
```

Typically, the gates that can be directly implemented in hardware will act only on one or two
qubits. In our circuits, we may want to use complex gates that act on a great number of qubits.
Fortunately, this will not be a problem. With the one and two qubit gates given to us by the
hardware, it is possible to build any other gate.

In this chapter we will first introduce the most basic multi-qubit gates, as well as the mathematics
used to describe and analyse them. Then we'll show how to prove that these gates can be used to
create any possible quantum algorithm. The chapter then concludes by looking at small-scale uses of
quantum gates. For example, we see how to build three-qubit gates like the Toffoli from single- and
two-qubit operations.

::: figure

    img(src="images/toffoli.png")

{.caption} This is a Toffoli with 3 qubits(q0,q1,q2) respectively. In this circuit example, q0 is
connected with q2 but q0 is not connected with q1.

:::


----------------------------------------------------------------------------------------------------


## 2.2 Multiple Qubits and Entangled States


### 第一讲：方程组的几何解释

我们从求解线性方程组来开始这门课，从一个普通的例子讲起：方程组有$2$个未知数，一共有$2$个方程，分别来看方程组的“行图像”和“列图像”.

有方程组
```latex
\begin{cases}2x&-y&=0\\-x&+2y&=3\end{cases},
```
写作矩阵形式有

```latex
\begin{bmatrix}2&-1\\-1&2\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix},
```
通常我们把第一个矩阵称为系数矩阵$A$，将第二个矩阵称为向量$x$，将第三个矩阵称为向量$b$，于是线性方程组可以表示为$Ax=b$.

我们来看行图像，即直角坐标系中的图像：


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

x = [-2, 2, -2, 2]
y = [-4, 4, 0.5, 2.5]

fig = plt.figure()
plt.axhline(y=0, c='black')
plt.axvline(x=0, c='black')

plt.plot(x[:2], y[:2], x[2:], y[2:])

plt.draw()
```

    img(src="images/chapter01_1_0.png")



```python
plt.close(fig)
```

上图是我们都很熟悉的直角坐标系中两直线相交的情况，接下来我们按列观察方程组

```latex
x\begin{bmatrix}2\\-1\end{bmatrix}+y\begin{bmatrix}-1\\2\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}.
```

（我们把第一个向量称作$col_1$，第二个向量称作$col_2$，以表示第一列向量和第二列向量），要使得式子成立，需要第一个向量加上两倍的第二个向量，即

```latex
1\begin{bmatrix}2\\-1\end{bmatrix}+2\begin{bmatrix}-1\\2\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}.
```

现在来看列图像，在二维平面上画出上面的列向量：


```python
from functools import partial

fig = plt.figure()
plt.axhline(y=0, c='black')
plt.axvline(x=0, c='black')
ax = plt.gca()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-3, 4)

arrow_vector = partial(plt.arrow, width=0.01, head_width=0.1, head_length=0.2, length_includes_head=True)

arrow_vector(0, 0, 2, -1, color='g')
arrow_vector(0, 0, -1, 2, color='c')
arrow_vector(2, -1, -2, 4, color='b')
arrow_vector(0, 0, 0, 3, width=0.05, color='r')

plt.draw()
```

    img(src="images/chapter01_4_0.png")




```python
plt.close(fig)
```

如图，绿向量$col_1$与蓝向量（两倍的蓝绿向量$col_2$）合成红向量$b$.

接着，我们继续观察

```latex
x\begin{bmatrix}2\\-1\end{bmatrix}+y\begin{bmatrix}-1\\2\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}
```

，$col_1,col_2$的某种线性组合得到了向量$b$，那么$col_1,col_2$的所有线性组合能够得到什么结果？它们将铺满整个平面.

下面进入三个未知数的方程组：

```latex
\begin{cases}2x&-y&&=0\\-x&+2y&-z&=-1\\&-3y&+4z&=4\end{cases},
```
写作矩阵形式

```latex
A=\begin{bmatrix}2&-1&0\\-1&2&-1\\0&-3&4\end{bmatrix},\ b=\begin{bmatrix}0\\-1\\4\end{bmatrix}.
```

在三维直角坐标系中，每一个方程将确定一个平面，而例子中的三个平面会相交于一点，这个点就是方程组的解.

同样的，将方程组写成列向量的线性组合，观察列图像：

```latex
x\begin{bmatrix}2\\-1\\0\end{bmatrix}+y\begin{bmatrix}-1\\2\\-3\end{bmatrix}+z\begin{bmatrix}0\\-1\\4\end{bmatrix}=\begin{bmatrix}0\\-1\\4\end{bmatrix}.
```

易知教授特意安排的例子中最后一个列向量恰巧等于等式右边的$b$向量，所以我们需要的线性组合为$x=0,y=0,z=1$.假设我们令

```latex
b=\begin{bmatrix}1\\1\\-3\end{bmatrix},
```
则需要的线性组合为$x=1,y=1,z=0$.

我们并不能总是这么轻易的求出正确的线性组合，所以下一讲将介绍消元法——一种线性方程组的系统性解法.

现在，我们需要考虑，对于任意的$b$，是否都能求解$Ax=b$？用列向量线性组合的观点阐述就是，列向量的线性组合能否覆盖整个三维向量空间？对上面这个例子，答案是肯定的，这个例子中的$A$是我们喜欢的矩阵类型，但是对另一些矩阵，答案是否定的.那么在什么情况下，三个向量的线性组合得不到$b$？

——如果三个向量在同一个平面上，问题就出现了——那么他们的线性组合也一定都在这个平面上.举个例子，比如$col_3=col_1+col_2$，那么不管怎么组合，这三个向量的结果都逃不出这个平面，因此当$b$在平面内，方程组有解，而当$b$不在平面内，这三个列向量就无法构造出$b$.在后面的课程中，我们会了解到这种情形称为**奇异**、**矩阵不可逆**.

下面我们推广到九维空间，每个方程有九个未知数，共九个方程，此时已经无法从坐标图像中描述问题了，但是我们依然可以从求九维列向量线性组合的角度解决问题，仍然是上面的问题，是否总能得到$b$？当然这仍取决于这九个向量，如果我们取一些并不相互独立的向量，则答案是否定的，比如取了九列但其实只相当于八列，有一列毫无贡献（这一列是前面列的某种线性组合），则会有一部分$b$无法求得.

接下来介绍方程的矩阵形式$Ax=b$，这是一种乘法运算，举个例子，取

```latex
A=\begin{bmatrix}2&5\\1&3\end{bmatrix},\ x=\begin{bmatrix}1\\2\end{bmatrix},
```
来看如何计算矩阵乘以向量：

* 我们依然使用列向量线性组合的方式，一次计算一列，

```latex
\begin{bmatrix}2&5\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix}=1\begin{bmatrix}2\\1\end{bmatrix}+2\begin{bmatrix}5\\3\end{bmatrix}=\begin{bmatrix}12\\7\end{bmatrix}
```

* 另一种方法，使用向量内积，矩阵第一行向量点乘$x$向量
```latex
\begin{bmatrix}2&5\end{bmatrix}\cdot\begin{bmatrix}1&2\end{bmatrix}^T=12,\ \begin{bmatrix}1&3\end{bmatrix}\cdot\begin{bmatrix}1&2\end{bmatrix}^T=7
```


教授建议使用第一种方法，将$Ax$看做$A$列向量的线性组合.



----------------------------------------------------------------------------------------------------


## 2.3 Phase Kickback


### 第二讲：矩阵消元

这个方法最早由高斯提出，我们以前解方程组的时候都会使用，现在来看如何使用矩阵实现消元法.

#### 消元法

有三元方程组

```latex
\begin{cases}x&+2y&+z&=2\\3x&+8y&+z&=12\\&4y&+z&=2\end{cases}$，对应的矩阵形式$Ax=b$为$\begin{bmatrix}1&2&1\\3&8&1\\0&4&1\end{bmatrix}\begin{bmatrix}x\\y\\z\end{bmatrix}=\begin{bmatrix}2\\12\\2\end{bmatrix},
```

按照我们以前做消元法的思路：

* 第一步，我们希望在第二个方程中消去$x$项，来操作系数矩阵

```latex
A=\begin{bmatrix}\underline{1}&2&1\\3&8&1\\0&4&1\end{bmatrix},
```

下划线的元素为第一步的主元（pivot）：

```latex
\begin{bmatrix}\underline{1}&2&1\\3&8&1\\0&4&1\end{bmatrix}\xrightarrow{row_2-3row_1}\begin{bmatrix}\underline{1}&2&1\\0&2&-2\\0&4&1\end{bmatrix},
```

这里我们先不管$b$向量，等做完$A$的消元可以再做$b$的消元.（这是MATLAB等工具经常使用的算法.）
* 第二步，我们希望在第三个方程中消去$y$项，现在第二行第一个非零元素成为了第二个主元：

```latex
\begin{bmatrix}\underline{1}&2&1\\0&\underline{2}&-2\\0&4&1\end{bmatrix}\xrightarrow{row_3-2row_2}\begin{bmatrix}\underline{1}&2&1\\0&\underline{2}&-2\\0&0&\underline{5}\end{bmatrix}
```
    
注意到第三行消元过后仅剩一个非零元素，所以它就成为第三个主元.做到这里就算消元完成了.

再来讨论一下消元失效的情形：首先，主元不能为零；其次，如果在消元时遇到主元位置为零，则需要交换行，使主元不为零；最后提一下，如果我们把第三个方程$z$前的系数成$-4$，会导致第二步消元时最后一行全部为零，则第三个主元就不存在了，至此消元不能继续进行了，这就是下一讲中涉及的不可逆情况.

* 接下来就该回代（back substitution）了，这时我们在$A$矩阵后面加上$b$向量写成增广矩阵（augmented matrix）的形式：

```latex
\left[\begin{array}{c|c}A&b\end{array}\right]=\left[\begin{array}{ccc|c}1&2&1&2\\3&8&1&12\\0&4&1&2\end{array}\right]\to\left[\begin{array}{ccc|c}1&2&1&2\\0&2&-2&6\\0&4&1&2\end{array}\right]\to\left[\begin{array}{ccc|c}1&2&1&2\\0&2&-2&6\\0&0&5&-10\end{array}\right].
```

不难看出，$z$的解已经出现了，此时方程组变为

```latex
\begin{cases}x&+2y&+z&=2\\&2y&-2z&=6\\&&5z&=-10\end{cases},
```

从第三个方程求出$z=-2$，代入第二个方程求出$y=1$，在代入第一个方程求出$x=2$.

#### 消元矩阵

上一讲我们学习了矩阵乘以向量的方法，有三个列向量的矩阵乘以另一个向量，按列的线性组合可以写作

```latex
\Bigg[v_1\ v_2\ v_3\Bigg]\begin{bmatrix}3\\4\\5\end{bmatrix}=3v_1+4v_2+5v_3.
```

但现在我们希望用矩阵乘法表示行操作，则有

```latex
\begin{bmatrix}1&2&7\end{bmatrix}\begin{bmatrix}&row_1&\\&row_2&\\&row_3&\end{bmatrix}=1row_1+2row_2+7row_3.
```

易看出这里是一个行向量从左边乘以矩阵，这个行向量按行操作矩阵的行向量，并将其合成为一个矩阵行向量的线性组合.

介绍到这里，我们就可以将消元法所做的行操作写成向量乘以矩阵的形式了.

* 消元法第一步操作为将第二行改成$row_2-3row_1$，其余两行不变，则有

```latex
\begin{bmatrix}1&0&0\\-3&1&0\\0&0&1\end{bmatrix}\begin{bmatrix}1&2&1\\3&8&1\\0&4&1\end{bmatrix}=\begin{bmatrix}1&2&1\\0&2&-2\\0&4&1\end{bmatrix}.
```

（另外，如果三行都不变，消元矩阵就是单位矩阵

```latex
I=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix},
```


$I$之于矩阵运算相当于$1$之于四则运算.）这个消元矩阵我们记作$E_{21}$，即将第二行第一个元素变为零.

* 接下来就是求$E_{32}$消元矩阵了，即将第三行第二个元素变为零，则

```latex
\begin{bmatrix}1&0&0\\0&1&0\\0&-2&1\end{bmatrix}\begin{bmatrix}1&2&1\\0&2&-2\\0&4&1\end{bmatrix}=\begin{bmatrix}1&2&1\\0&2&-2\\0&0&5\end{bmatrix}
.
```

这就是消元所用的两个初等矩阵（elementary matrix）.

* 最后，我们将这两步综合起来，即$E_{32}(E_{21}A)=U$，也就是说如果我们想从$A$矩阵直接得到$U$矩阵的话，只需要$(E_{32}E_{21})A$即可.注意，矩阵乘法虽然不能随意变动相乘次序，但是可以变动括号位置，也就是满足结合律（associative law），而结合律在矩阵运算中非常重要，很多定理的证明都需要巧妙的使用结合律.

既然提到了消元用的初等矩阵，那我们再介绍一种用于置换两行的矩阵：置换矩阵（permutation matrix）：例如

```latex
\begin{bmatrix}0&1\\1&0\end{bmatrix}\begin{bmatrix}a&b\\c&d\end{bmatrix}=\begin{bmatrix}c&d\\a&b\end{bmatrix},
```

置换矩阵将原矩阵的两行做了互换.顺便提一下，如果我们希望交换两列，则有

```latex
\begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix}=\begin{bmatrix}b&a\\d&c\end{bmatrix}.
```

我们现在能够将$A$通过行变换写成$U$，那么如何从$U$再变回$A$，也就是求消元的逆运算.对某些“坏”矩阵，并没有逆，而本讲的例子都是“好”矩阵.

#### 逆

现在，我们以$E_{21}$为例，

```latex
\Bigg[\quad ?\quad \Bigg]\begin{bmatrix}1&0&0\\-3&1&0\\0&0&1\end{bmatrix}=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix},
```

什么矩阵可以取消这次行变换？这次变换是从第二行中减去三倍的第一行，那么其逆变换就是给第二行加上三倍的第一行，所以逆矩阵就是

```latex
\begin{bmatrix}1&0&0\\3&1&0\\0&0&1\end{bmatrix}.
```

我们把矩阵$E$的逆记作$E^{-1}$，所以有$E^{-1}E=I$.



----------------------------------------------------------------------------------------------------


## 2.4 More Circuit Identities

TODO


----------------------------------------------------------------------------------------------------


## 2.5 Proving Universality

TODO
