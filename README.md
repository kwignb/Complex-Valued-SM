# Complex-Valued Subspace Method
入力を複素数としたときにどのような効果が得られ，また精度が上昇するか，下降するかを確認するために弊研究室でよく取り組まれている部分空間法について複素数対応の実装を行った．

## 実装したもの　
- 複素相互部分空間法 (Complex-Valued Mutual Subspace Method)

## いずれ実装したい 
- 複素カーネル相互部分空間法 (Complex-Valued Kernel Mutual Subspace Method) \
カーネルの複素数対応が面倒そう？

## 変更点
- 入力を実数ではなく複素数にする
  - 変換方法は様々で特に決まってはいない
    - フーリエ変換 (Fourier Transformation)
    - ヒルベルト変換 (Hilbert Transformation)
    - 正規分布から生成した値を係数とする虚数を足す
    
- 分散共分散行列は共役転置により生成する
  - <img src="https://latex.codecogs.com/gif.latex?E[(Z-\bm{{\mu}})(Z-\bm{{\mu}})^{*}]" />

- 入力が複素数になることによって分散共分散行列が対称行列ではなくエルミート行列になる
  - 固有値問題を解くとき，エルミート行列は得られる固有値は実数
  - 固有ベクトルは複素ベクトル
  
- cos類似度は複素空間における内積の定義，すなわちエルミート内積を用いる
  - <img src="https://latex.codecogs.com/gif.latex?\langle\boldsymbol{x},\boldsymbol{y}\rangle&space;:=&space;\bar{\boldsymbol{x}}^{\mathrm{T}}\boldsymbol{y}&space;=&space;\sum_{i=1}^n&space;\bar{x_{i}}y_i"/>

## データセット
- [ETH-80](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/object-recognition-and-scene-understanding/analyzing-appearance-and-contour-based-methods-for-object-categorization/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

## 参考文献
- 福井和広, “部分空間法と識別器”, 第13回画像センシングシンポジウム, 2007.
- 福井和広, “最近の技術動向：相互部分空間法への拡張とその応用事例”, 情報処理学会, Vol.49, No.6, June 2008.
