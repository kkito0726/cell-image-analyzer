# 顕微鏡画像解析ライブラリ


## 構造テンソルによる配向解析 (Structure Tensor Analysis) 

構造テンソルは、画像中の各ピクセルにおける局所的な配向（向き）を検出するための強力な手法です。画像中のエッジやテクスチャの方向を定量化するために広く用いられます。

計算は、主に以下のステップで実行されます。

### 1. 画像勾配の計算

はじめに、画像の各ピクセルにおける輝度変化の方向と大きさを捉えるため、画像全体の輝度 $I$ に関するx方向とy方向の偏微分（勾配）を計算します。

$$ 
I_x = \frac{\partial I}{\partial x} 
$$

$$ 
I_y = \frac{\partial I}{\partial y} 
$$

この計算には、Sobelオペレータのような微分フィルタが一般的に使用されます。$I_x$ は水平方向の輝度変化を、$I_y$ は垂直方向の輝度変化を表します。

### 2. 構造テンソル行列の構築

次に、計算した勾配から各ピクセルごとに2x2の構造テンソル行列 $T$ を構築します。この行列は、勾配ベクトルの外積から作られます。

$$ T = 
\begin{bmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{bmatrix}
$$ 

### 3. 局所的な平均化

ノイズの影響を低減し、より安定した配向情報を得るために、構造テンソル行列の各要素を局所的なウィンドウ（近傍領域）で平均化します。通常、ガウシアンウィンドウ $w$ を用いた畳み込み演算が適用されます。

平均化された構造テンソル行列を $J$ とすると、各要素は以下のように計算されます。

$$ 
J = 
\begin{bmatrix}
\langle I_x^2 \rangle & \langle I_x I_y \rangle \\
\langle I_x I_y \rangle & \langle I_y^2 \rangle
\end{bmatrix}
= 
\begin{bmatrix}
J_{xx} & J_{xy} \\
J_{xy} & J_{yy}
\end{bmatrix}
$$ 

ここで、$\langle \cdot \rangle$ はガウシアンウィンドウによる重み付き平均を表します。
- $J_{xx} = w * I_x^2$
- $J_{yy} = w * I_y^2$
- $J_{xy} = w * (I_x I_y)$

### 4. 配向角の計算

最後に、平均化されたテンソル行列 $J$ の成分を用いて、配向角 $\theta$ を計算します。この角度は、画像の構造が最も顕著でない方向（勾配が最小になる方向）、すなわちエッジやテクスチャの接線方向に対応します。

配向角 $\theta$ は以下の式で与えられます。

$$ 
\theta = \frac{1}{2} \arctan 
\left( \frac{2 J_{xy}}{J_{xx} - J_{yy}} \right)
$$ 

`arctan2` 関数を用いると、`J_{xx} - J_{yy}` がゼロになる場合も安全に計算できます。

$$ 
\theta = \frac{1}{2} \operatorname{arctan2} (2 J_{xy}, J_{xx} - J_{yy})
$$ 

この $\theta$ は、画像中の構造（例：細胞の長軸）の配向角度を示します。値は通常、ラジアン単位で得られます。

### 5. 配向秩序パラメータ (Orientation Order Parameter)

配向秩序パラメータ（$S$）は、画像全体または特定の領域内において、構造（例：細胞や線維）がどの程度同じ方向を向いて整列しているかを示す指標です。

各ピクセルで計算された配向角 $\theta$ を用いて、以下の式で計算されます。

$$ 
S = \sqrt{\langle \cos(2\theta) \rangle^2 + \langle \sin(2\theta) \rangle^2}
$$ 

ここで、$\langle \cdot \rangle$ は、対象とする領域全体の平均を意味します。

#### パラメータの意味
配向秩序パラメータ $S$ は、0から1の間の値を取ります。
- **$S \approx 0$**: 配向が完全にランダム（バラバラ）な状態。特定の配向方向は存在しない。
- **$S \approx 1$**: 全ての構造が完全に同じ方向を向いて整列している状態。

この単一のスカラ値によって、画像内の構造の全体的な配向性を定量的に評価することができます。

--- 

## サンプルコード

### 細胞の配向角計算
```python
from pyCell import *

# 画像の読み込み
img = read_img("./sample.jpeg")

# CLAHEでコントラスト強調
img_eq = img.clahe()

# 軽い平滑化
img_blur = img_eq.gaussian_blur()

# 勾配
sobelXY = sobel_xy_factory(img_blur)

# 構造テンソル
tensor = structure_tensor_factory(sobelXY)

# ローズダイヤヒストグラム
tensor.rose_hist()

# 秩序パラメータ
print(tensor.orientation_order_parameter)
```