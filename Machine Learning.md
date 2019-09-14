# [Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome)

## 线性代数

$\begin{cases}矩阵加法\\矩阵乘法\\单位矩阵&I_{n*n}=eye(n)\\逆矩阵、伪逆矩阵&A^{-1}=inv(A)或pinv(A)\\转置矩阵&A^{T}=A'\end{cases}$

## Octave

$\begin{cases}系统\begin{cases}PS1('>>')&隐藏版本信息\\clear&清空变量\\disp(a)&只输出a的值，不输出a\\disp(sprintf("\%d",a))&C语言格式输出\\;&只赋值不输出变量值\\hist(a,n)&画出a的n条直方图\\plot(x,y,'color')&绘制图像\\hold\ on&保留旧图像\\legend('\ ')&增加图例\\axis([a\ b\ c\ d])&改刻度\\print\ -dpng\ 'name'&存图像\\subplot(1,n,x)&同时显示n个图，当前编辑第x个\\pwd&当前路径\\load&载入数据\\save\ a.bat v&把v存入文件\\who&显示变量\\whos&变量详细信息\\while\ a<3,……end;&while循环，for，if同\\\%XXX&注释\\find(y>1)&返回下标\end{cases}\\语法\begin{cases}sim=&不等于\\xor(1,0)&异或\\sqrt()&开方\\a'&转置矩阵\\pinv(a)\ inv(a)&伪逆矩阵、逆矩阵\\A=[1\ 2;3\ 4;5\ 6]&矩阵，三行两列\\V=1:0.2:2&矩阵，从1遍历到2，步长可省略\\ones(2,3)&矩阵，两行三列，全为1\\5*ones(2,3)&矩阵，两行三列，全为5\\zeros(2,3)&零矩阵\\rand(2,3)&随机矩阵，取值[0,1]\\randn(2,3)&随机矩阵，均值0，方差1\\eye(5)&五阶单位矩阵\\size(a,1/2)&a的行列数\\length(a)&最大维度\\a(3,2)&取a的三行二列\\a(3,:)&取a的三行全部\\a(:)&把a所有元素放入一个列向量\\ [a\ b]&矩阵合并\\a.*b&按位相乘\\std(X)&求标准差\\mean(X)&求平均值\\ [U, S, V]=svd(sigma)求特征向量\end{cases}\end{cases}$

## 机器学习

$有无数据标注\begin{cases}监督学习\begin{cases}预测：线性回归\begin{cases}样本数&m\\特征数&n\\样本1的特征2&x^{(1)}_2\\假设函数&h_\Theta(x)=\Theta_0+\Theta_1*x_1+……=\Theta^TX\begin{cases}梯度下降算法&\Theta_j:=\Theta_j-\alpha\frac{\partial}{\partial\Theta_j}J(\Theta)=\Theta_j-\alpha\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}&\begin{cases}特征缩放&x_i->\frac{x_i}{S_i}\\均值归一化&x_i->x_i-\mu_i\\代价函数&J(\Theta)=\frac{1}{2m}\sum^m_{i=1}(h_\Theta(x^{(i)})-y^{(i)})^2\end{cases}\\正规方程法&\Theta=(X^TX)^{-1}X^TY&X_{m*(n+1)}=\left[\begin{matrix}(X^{(1)})^T\\(X^{(2)})^T\\……\end{matrix}\right] \end{cases}\end{cases}\\分类\begin{cases}n>>m\ or\ n<< m \ 逻辑回归\begin{cases}h_\Theta(x)=1&\Theta^TX>=0\\h_\Theta(x)=0&\Theta^TX<0\\决策边界&\Theta^TX=0\\假设函数&h_\Theta(x)=P\{y=1|x;\theta\}=\frac{1}{1+e^{-\Theta^TX}}\begin{cases}梯度下降算法&\Theta_j:=\Theta_j-\alpha\frac{\partial}{\partial\Theta_j}J(\Theta)=\Theta_j-\alpha\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}\begin{cases}特征缩放&x_i->\frac{x_i}{S_i}\\均值归一化&x_i->x_i-\mu_i\\代价函数&J(\Theta)=\frac{1}{m}\sum_{i=1}^{m}[-ylog(h_{\theta}(x))-(1-y)log(1-h_{\theta}(x))]\end{cases}\\共轭梯度法\\变尺度法\\限制变尺度法\end{cases}\\多分类器&拆分成多个h^{(i)}_{\theta}(x)\\无法用直线表示&增加\theta，防止欠拟合\end{cases}\\无论nm比例\ 神经网络\\n<m\ SVM\end{cases}\\过拟合\begin{cases}删除特征\\正则化&\begin{cases}线性回归\begin{cases}\\梯度下降法&\begin{cases}J(\Theta)=\frac{1}{2m}(\sum^m_{i=1}(h_\Theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{i=1}^{n}\theta_j^2)\\\Theta_j:=\Theta_j-\alpha\frac{\partial}{\partial\Theta_j}J(\Theta)=\Theta_j-\alpha(\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j)=\Theta_j(1-\alpha\frac{\lambda}{m})-\alpha\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}\end{cases}\\正规方程法&\Theta=(X^TX+\left[\begin{matrix}000\\010\\001\\……\end{matrix}\right])^{-1}X^TY\end{cases}\\逻辑回归J(\Theta)=\frac{1}{m}\sum_{i=1}^{m}[-ylog(h_{\theta}(x))-(1-y)log(1-h_{\theta}(x))]+\frac{\lambda}{2m}\sum_{i=1}^{n}\theta_j^2\end{cases}\end{cases}\end{cases}\\无监督学习\begin{cases}聚类算法\\异常检测算法\\……\end{cases}\end{cases}$

1. 假设函数：表示预测值，向量$\Theta$参数

2. 梯度下降算法：$\alpha​$表示学习速率，j取0到n同步更新直到收敛，**特征数n在10000以上用**

3. 特征缩放：把特征范围缩放到[-3，3]到[$-\frac{1}{3}$,$\frac{1}{3}$]左右，$S_i$为max-min

4. 均值归一化：把特征范围归一到以0为中心，$\mu_i$为平均值

5. 代价函数：表示误差，寻找向量$\Theta$使得$J(\Theta)$取最小值

6. 正规方程法：一步到位，**特征数n在10000以下用（$X^TX$运算量过大）**，当m小于n可能导致矩阵不可逆，应先删除多余特征

7. 正则化：当$\theta$较多时，给代价函数加上很大的正规化参数$\lambda$乘每个$\theta$，作为惩罚项，以放大每个$\theta$的影响，从而减小$\theta$的值，防止过拟合。$\theta_0$恒为1，不需惩罚。

## Exercise

1. 先做样例可视化， 算完后画学习曲线进行误差分析

2. 同步更新所有$\theta_j​$

3. Octave中从1开始

4. 找学习速率技巧：每次乘除3

5. 库函数fminunc：提供cost函数，返回$\theta$和cost

    ``` Matlab

    options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

    ```
6. 正规化逻辑回归时，不惩罚$\theta_0$

## 神经网络

1. 输入层（第1层）->隐藏层->输出层（$h_\theta(x)$）

2. 总层数L，第l层神经元数$S_l$(不含偏置单元)

3. 权重$\theta^{(j)}$表示j层对j+1层的权重，$dim(\theta^{(i)})=s_{j+1}*(s_j+1)$

4. 随机初始化，如10*11向量：

   ``` Matlab

   THETA=rand(10,11)*(2*INIT_EPSILON)-INIT_EPSILON;

   ```

5. 正向传播算法

   第j层i个神经元激励值$a^{(j)}_i=g(z^{(j)}_i)=g(\theta^{j-1}_{i0}a_0^{j-1}+\theta^{j-1}_{i1}a_1^{j-1}+……),g(x)=\frac{1}{1+e^{-\theta x}}$

   第j层所有神经元激励值$a^{(j)}=g(z^{(j)})=g(\theta^{(j-1)}X)$

6. 代价函数: $J(\theta)=-\frac{1}{m}[\sum^{m}_{i=1}\sum^{K}_{k=1}y_k^{(i)}log(h_{\theta}(x^{(i)}))_k+(1-y_k^{(i)})log(1-(h_{\theta}(x^{(i)}))_k)]+\frac{\lambda}{2m}\sum^{L-1}_{l=1}\sum^{s_l}_{i=1}\sum^{s_{l+1}}_{j=1}(\theta^{(l)}_{ji})^2$

7. 反向传播算法

   $\delta^{(l)}_j$第l层j个神经元的误差$\begin{cases}\delta^{(l)}_j=a^{(l)}_j-y_j&l=L\\\delta^{(l)}_j=(\Theta^{(l)})^T\delta^{(l+1)}.*a^{(l)}.*(1-a^{(l)})&1< l<L\end{cases}$

   误差向量$\Delta，有\Delta^{(l)}+=\delta^{(l+1)}*(a^{(l)})^T$

   $\frac{\partial}{\partial\Theta^{(l)}_{ij}}J(\Theta)=a^{(l)}_j\delta^{(l+1)}_i=D_{ij}^{(l)}=\begin{cases}\frac{1}{m}\Delta_{ij}^{(l)}+\frac{\lambda}{m}\Theta_{ij}^{(l)}&j\ne0\\\frac{1}{m}\Delta_{ij}^{(l)}&j=0\end{cases}$

8. 梯度检验$\begin{cases}计算反向传播\\计算梯度检验&\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}\\比较\\关闭梯度检验\end{cases}$

9. 隐藏层选择：一般1个隐藏层，多个时，每个隐藏层相同神经元个数。神经元个数越多越好，但过多时计算量过大。

10. 神经网络训练较慢，但对n和m各种比例表现都很好

## SVM支持向量机

1. 核函数：将每个训练样本作为一个选定点，取m个特征变量$f_1,f_2……$

   $f_i=exp(-\frac{||x-l^{(i)}||^2}{2\sigma^2})$

   此时预测值$h=\theta_0+\theta_1f_1+\theta_2f_2+\theta_3f_3+……$，即该点到三个选定点的远近，h>=0则预测y=1

2. $J = C\sum^m_{i=1}[y^{(i)}cost_1(\theta^Tf^{(i)})+(1-y^{(i)}cost_0(\theta^Tf^{(i)}))]+\frac{1}{2}\sum^m_{i=1}\theta^2$

   C近似于$\frac{1}{\lambda}$

3. 特征数n小，样本数m大时，svm表现好

## 聚类算法

K均值算法：选取聚类数K（尝试一系列K，取J下降最快的K；或根据问题手动选择K），随机化K个聚类中心$\mu_1,\mu_2……\mu_K$（选K个样本作为初始中心，选100次，取J最小一次），重复簇分配（把样本分配给最近的簇中心）、移动聚类中心（以该中心所属样本位置的平均值作为移动位置）直到分为K个簇。$c^{(i)}表示样本x^{(i)}被分到第几个簇中，其聚类中心为\mu_{c^{(i)}}$

## 异常检测

把训练集作为正常样本，根据测试集P于阈值的大小，预测测试集是否正常。要求在训练集异常样本少时使用，即对正常样本建模。异常样本和正常样本都多时，可以用监督算法，即对异常样本建模。

1. 异常检测算法：

   选择能体现反常的特征$x_i$

   处理数据，令其类似高斯分布，如取log(x+c)、$x^c$等方式

   计算$\mu_1、\mu_2……\mu_i=\frac{1}{m}\Sigma^m_{i=1}x^{(i)}_j$

   计算$\sigma_1^2、\sigma_2^2……\sigma^2_j=\frac{1}{m}(x^{(i)}_j-\mu_j)^2$

   计算新样本的高斯分布$P(x)=\Pi^n_{j=1}p(x_j;\mu_j;\sigma_j)=\Pi^n_{j=1}\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x_j-\mu_j)^2}{2\sigma^2})<\epsilon$则为异常

   $\epsilon可根据F最大时的\epsilon来选择$

## 机器学习诊断法

1. 诊断过拟合：将训练集取出30%作为测试集$\begin{cases}J_{train}(\theta)很小但差分率J_{test}(\theta)很大则过拟合\\0/1错分率很大& error=1\begin{cases}y=0\ and\ h(\theta)>=0.5\\y=1\ and\ h(\theta)<0.5\end{cases}\end{cases}$

2. 诊断欠拟合：$J_{train}(\theta)和J_{test}(\theta)都很大，二者基本相近$

3. 模型选择问题：$确定多项式次数\ 数据集分为训练集:交叉验证集:测试集=6:2:2，求出各个预测函数的\theta后，求J_{CV}(\theta)最小的预测函数$

   $J(\theta)=\frac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2+\frac{\lambda}{2m}\sum^m_{j=1}\theta^2_j$

   $J_{train}(\theta)=\frac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2$

   $J_{cv}(\theta)=\frac{1}{2m_{cv}}\sum^{m_{cv}}_{i=1}(h_{\theta}(x_{cv}^{(i)})-y_{cv}^{(i)})^2$

4. 学习曲线：$J_{train}(\theta)和J_{cv}(\theta)关于m的h函数$

   高偏差（欠拟合）：$J_{train}(\theta)和J_{cv}(\theta)$已经水平，m增加$J_{train}(\theta)和J_{cv}(\theta)$并不下降

   高方差（过拟合）：$J_{train}(\theta)和J_{cv}(\theta)$较远，m增加将汇合

5. 查准率Precision：预测为1中真正为1的比例

6. 召回率Recall：真正为1中预测为1的比例

7. 偏斜问题：分类问题时训练集比例悬殊，以查准率和召回率来评估，$F=\frac{2PR}{P+R}$越大算法越好

8. $F_\beta=\frac{(1+\beta^2)P* R}{\beta^2*P+R}$用$\beta$作为权重，根据实际P和R哪个重要调整$\beta$

## 维数约减

主成分分析法：特征缩放、均值归一化后，计算协方差矩阵$\Sigma=\frac{1}{m}(x^{(i)})(x^{(i)})^T$，求$\Sigma$的特征向量U，其前K列转置乘X得到Z为约减后的样本，每个样本只有k个特征

选取K：$1-\frac{\Sigma^k_{i=1}S_{ii}}{\Sigma^n_{i=1}S_{ii}}$<=0.01，则为保留99%的差异性

主成分分析法可加速算法，减少数据大小，但并不用来避免过拟合