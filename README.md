# Complex-Valued Subspace Method
入力を複素数としたときにどのような効果が得られ，また精度が上昇するか，下降するかを確認するために弊研究室で多く取り組まれている部分空間法について複素数対応の実装を行った．

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
    
- 入力が複素数になることによって分散共分散行列が対称行列ではなくエルミート行列になる
  - 固有値問題を解くとき，エルミート行列は得られる固有値は実数
  - 固有ベクトルは複素ベクトル
  
- cos類似度は複素空間における内積の定義を用いる

## Dataset
- ETH-80
- CIFAR10
