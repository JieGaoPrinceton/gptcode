# Math

## Matrix

Casorati 矩阵是一种用于处理多维数据的重构方法，常用于信号处理、图像处理和其他涉及多维数据分析的领域。其目的是将多维数据展平，形成一个矩阵，使得可以使用诸如奇异值分解（SVD）等线性代数方法对其进行分析。

### Casorati 矩阵的特点：

* **多维数据展平**：Casorati 矩阵将一个高维的数据块（例如3D或更高维的张量）转换为二维矩阵，使得后续的数学运算更容易进行。
* **时间维度或特征维度**：在将数据重构为 Casorati 矩阵时，通常会将时间维度或特征维度保留在矩阵的列中，而其他维度展平到行中。

在您的代码中，当多维数据被检测到时，将其重塑为 Casorati 矩阵的形式，即将所有维度展平（flatten），除了最后一个维度，例如：

* 如果原数据是形状为 `(a, b, c)` 的 3D 数组，经过重塑后，Casorati 矩阵的形状会变为 `(a*b, c)`，其中最后一个维度被保留，而前面的维度被展平。

这种转换的好处在于，您可以方便地对多维数据进行矩阵操作，例如奇异值分解（SVD）、主成分分析（PCA）等，来提取数据的特征或进行降维。

# 评估方法

在程序评估中，常用的指标包括 **RMSE**、**Jaccard**、**精度**、**Gap** 和 **时间**，每个指标用于衡量不同的方面，帮助评估模型或算法的性能。以下是对这些指标的详细介绍：

### 1. RMSE（Root Mean Square Error，均方根误差）

**RMSE** 是一种常用的衡量误差的指标，尤其适用于回归模型，用于评估预测值与实际值之间的差距。它的计算公式为：

\[
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
\]

其中：

- \( y_i \) 表示真实值
- \( \hat{y}_i \) 表示预测值
- \( N \) 表示样本数量

**RMSE** 越小，表示模型预测的精确度越高。它对较大的误差特别敏感，因为误差的平方会放大这些影响。

### 2. Jaccard 系数（Jaccard Index）

**Jaccard 系数** 是一种用于衡量两个集合相似度的指标，尤其常用于分类问题或聚类问题，评估两个集合的重合情况。它的计算公式为：

\[
\text{Jaccard} = \frac{|A \cap B|}{|A \cup B|}
\]

其中：

- \( A \) 和 \( B \) 分别为两个集合
- \( |A \cap B| \) 表示两个集合的交集
- \( |A \cup B| \) 表示两个集合的并集

**Jaccard 系数** 的取值范围是 0 到 1，值越大，两个集合的相似度越高。

### 3. 精度（Accuracy）

**精度** 是衡量分类器性能的一个常用指标，通常用于分类问题。精度是正确分类样本的数量与总样本数量的比值，其计算公式为：

\[
\text{精度} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
\]

其中：

- **TP**（True Positive）：正确分类为正类的数量
- **TN**（True Negative）：正确分类为负类的数量
- **FP**（False Positive）：错误分类为正类的数量
- **FN**（False Negative）：错误分类为负类的数量

精度的取值范围为 0 到 1，值越高，表示模型的分类准确率越高。

### 4. Gap

**Gap** 通常用于衡量算法在训练集和测试集之间性能差距的指标。它表示模型是否发生了**过拟合**或者**欠拟合**的情况。Gap 可以是指：

- **Loss Gap（损失差距）**：训练集损失和测试集损失之间的差距。
- **Accuracy Gap（精度差距）**：训练集和测试集的精度差异。

**Gap** 值越大，意味着模型可能在训练集上表现很好，但在测试集上表现不佳，存在过拟合的情况。因此，我们希望 **Gap** 越小越好，表示模型对未知数据的泛化能力较好。

### 5. 时间（Time）

**时间** 指的是算法或模型的执行时间，通常用于衡量模型的效率。时间可以包括多个部分，例如：

- **训练时间**：模型在训练集上进行训练所需的时间。
- **预测时间**：模型对新数据进行预测的时间。
- **总执行时间**：从训练到预测所需的总时间。

在实际应用中，时间指标对模型的选择至关重要，尤其是在实时性要求较高的应用场景中（如自动驾驶、金融交易等），时间越短越好。

### 全局评分

要计算程序的整体评估（全局评分），可以结合以上多个指标进行综合评价。通常可以使用加权平均或其他多指标融合的方法，将所有评估指标整合为一个全局评分。具体做法可能包括以下几种方式：

1. **加权评分**：为每个指标分配一个权重，并根据每个指标的得分来计算加权平均。
2. **归一化评分**：对所有指标进行归一化处理（例如，将它们的范围调整到 [0, 1] 之间），然后再进行加权组合。

这能帮助评估模型的整体性能，不仅关注模型的准确度，还考虑模型的时间效率、泛化能力等。通过结合不同的指标，能够更全面地评估模型在实际应用中的表现。


# 读取Raw Data的核心代码

```
IQfiles = dir([mydatapath '*.mat']);
Nbuffers = numel(IQfiles);
load([IQfiles(1).folder filesep IQfiles(1).name], 'UF', 'PData');
```


* 获取 IQ 文件列表 [`IQfiles`](vscode-file://vscode-app/c:/Users/ericg/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)。
* 计算文件数量 [`Nbuffers`](vscode-file://vscode-app/c:/Users/ericg/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)。
* 加载第一个 IQ 文件，提取 [`UF`](vscode-file://vscode-app/c:/Users/ericg/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 和 [`PData`](vscode-file://vscode-app/c:/Users/ericg/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 变量。

```
load([IQfiles(1).folder filesep IQfiles(1).name], 'IQ');
NFrames = size(IQ, 3);
PData.Origin = [0 PData.Size(2)/2*PData.PDelta(2) 0];
framerate = UF.FrameRateUF;
```


* 加载第一个 IQ 文件，提取 [`IQ`](vscode-file://vscode-app/c:/Users/ericg/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 数据。
* 计算帧数 [`NFrames`](vscode-file://vscode-app/c:/Users/ericg/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)。
* 设置 `PData.Origin` 参数。
* 获取帧率 [`framerate`](vscode-file://vscode-app/c:/Users/ericg/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html)。这个帧率，在mat文件的UF结构体中已经设定好了。

```ULM
'res',10,...                        % Resolution factor. Typically 10 for images at lambda/10.
'SVD_cutoff',[5 UF.NbFrames],...    % SVD filtering
'max_linking_distance',2,...        % Maximum linking distance between two frames to reject pairing, in pixels units (UF.scale(1)). (2-4 pixel).
'min_length', 15,...                % Minimum length of the tracks. (5-20)
'fwhm',[1 1]*3,...                  % Size of the mask for localization. (3x3 for pixel at lambda, 5x5 at lambda/2). [fwhmz fwhmx]
'max_gap_closing', 0,...            % Allowed gap in microbubbles pairing. (0)
'size',[PData.Size(1),PData.Size(2),UF.NbFrames],... % Size of the data [z, x, t]
'scale',[1 1 1/framerate],...       % Scale [z x t]
'numberOfFramesProcessed',UF.NbFrames,... % Number of processed frames
'interp_factor',1/res...            % Interpolation factor
);

```


### 字段解释：

1. **numberOfParticles**：
   * **值**：70
   * **解释**：每帧的粒子数量。通常在 30 到 100 之间。
2. **res**：
   * **值**：10
   * **解释**：分辨率因子。通常为 10，用于生成分辨率为 lambda/10 的图像。
3. **SVD\_cutoff**：
   * **值**：[5 UF.NbFrames]
   * **解释**：SVD 滤波的截止值。这里设置为从第 5 帧到总帧数 `UF.NbFrames`。
4. **max\_linking\_distance**：
   * **值**：2
   * **解释**：两帧之间的最大链接距离，以像素为单位（`UF.scale(1)`）。通常在 2 到 4 像素之间。
5. **min\_length**：
   * **值**：15
   * **解释**：轨迹的最小长度。通常在 5 到 20 之间。
6. **fwhm**：
   * **值**：[1 1]\*3
   * **解释**：用于定位的掩模大小。对于像素大小为 lambda 的图像，通常为 3x3；对于像素大小为 lambda/2 的图像，通常为 5x5。这里设置为 [3 3]。
7. **max\_gap\_closing**：
   * **值**：0
   * **解释**：微泡配对中允许的最大间隙。这里设置为 0。
8. **size**：
   * **值**：[PData.Size(1), PData.Size(2), UF.NbFrames]
   * **解释**：数据的大小，包括 z 方向、x 方向和时间方向的尺寸。
9. **scale**：
   * **值**：[1 1 1/framerate]
   * **解释**：比例因子，包括 z 方向、x 方向和时间方向的比例。时间方向的比例为 1/帧率。
10. **numberOfFramesProcessed**：
    * **值**：UF.NbFrames
    * **解释**：处理的帧数。
11. **interp\_factor**：
    * **值**：1/res
    * **解释**：插值因子。这里设置为 1/分辨率因子。
