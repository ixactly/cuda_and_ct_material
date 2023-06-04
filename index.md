<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# GPUによるプログラミング入門
## CUDAとは
**CUDA**(Compute Unified Device Architecture)はNVIDIA社によるGPUプログラミング開発環境です。通常、一般的なコンピュータはCPUを使用して計算を行いますが、 CUDAを使用することでGPUも計算に利用することができます。
GPUは、多数のコアと高い並列処理能力により、 **並列化が可能な演算**を高速に処理することができます。（単一の処理自体は速くないことに注意）
CUDAは、CやC++をベースとしたプログラミングモデルを提供しており、GPUを使用した並列計算を容易に実装することができます。GPUの性能を真に引き出して高速化したい場合にはハードウェアアーキテクチャに対する理解が必要になるため、難しいです（自分もあまり分かっていません）。逆に言うとハードウェアを意識したプログラミングも可能であると言うことです。  

## CUDAプログラミング
CUDAで出来ることは、単一の命令を与えて、複数の処理を行うことです。（SIMT: Single Instruction Multiple Thread , SPMD: Single Program Multiple Data など）  
そこでCUDAでは、C，C++で書かれた関数に対して並列化をかけることによって高速化を目指します。ここでは、並列化するための準備を行なっていきます。
### 1. CUDAとCの相違点
CUDAは基本的にはC言語とほとんど相違はありません。そのため、Cで書かれたプログラムに関してはそのまま動作すると思われます。
CUDAではCに対して大きく2つの拡張がなされています。
1. ホストメモリ（CPU）とデバイスメモリ（GPU）間でデータを転送する機能
2. ホスト（CPU）から、デバイス（GPU）に対して、多数のスレッドを並列に命令を与える機能  
   
CUDAのパラダイムでは、CPU側を**ホスト**、GPU側を**デバイス**と呼びます。1について、ホストとデバイスは異なるメモリシステムとなっているため、ホスト側からデバイスメモリを参照することは出来ません（逆も然り。厳密には異なるかもしれないが、大まかにはRAMがホストメモリ、VRAMがデバイスメモリと思ってもらって良い）。そのため、並列計算に必要となるボリュームデータや点群データなどの入力データはデバイスメモリに事前に転送して置く必要があります。また、計算後の出力データもデバイスからホストへと転送しなければなりません。2についてはそのままの意味で、並列計算の命令を与えます。基本的にはforループなどの、indexが変わるけれど、実行内容が変わらないような命令を並列化することになるでしょう。  

### 2. 準備
CUDAはNVIDIA社製のGPUにのみ対応しているため、GeForceやTesla, QuadroのGPUが必要になります。
### 2.1. インストール
[公式ページ](https://developer.nvidia.com/cuda-downloads)から自身が利用している環境を選択してインストーラーを入れましょう。([インストールガイド](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html))  
インストーラの指示に従ってインストールを行ってください。インストールが行われれば、`nvcc`というCUDAコンパイラが導入されているはずです。正しくインストールされたかどうかは`ncvv -V`というコマンドで確認してください。  
![nvcc_check](nvcc.png)  
ソースコードをコンパイルする場合は、`gcc`や`clang`と同様、`nvcc main.cu`のようなコマンドでコンパイルが可能です。CUDAで実装したソースコードの命名規則ですが、ソースファイルであれば`*.cu`、ヘッダファイルであれば`*.cuh`のように命名します。  
プロジェクトが大きくなってくると、IDEを用いた管理をすると楽でしょう。**CLion**や**Visual Studio**によるプロジェクト管理を行いましょう。CLionならば[こちら](https://pleiades.io/help/clion/cuda-projects.html)を、VSならば[こちら](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html?highlight=visual%20studio#build-customizations-for-existing-projects)を参考にしてみてください。



### 3. プログラミングの流れ
一般的なCUDAプログラムの構造は主に以下の5つのstepに分かれます。
1. GPUメモリの確保
2. CPUメモリからデータをGPUメモリにコピー
3. CUDAカーネルを呼び出し、並列計算を実行する
4. GPUメモリからデータをCPUメモリにコピー
5. GPUメモリの解放

実際に書くと以下のようなプログラムになるでしょう。ここでは与えた配列に対して全要素に1を足すというプログラムを例として示します。

```c++
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void plusOneArray(float* array) {
    unsigned int u = blockDim.x * blockIdx.x + threadIdx.x;
    array[u] += 1.0f;
    printf("block idx: %d, thread idx: %d, device_array[%d]: %f\n", blockIdx.x, u, u, array[u]);
}

int main() {
    int N = 1024;
    // init host memory
    float host_array[1024];
    for(int i = 0; i < N; i++)
        host_array[i] = (float)i;
    // init device memory
    float* device_array;

    // 1. allocate device memory
    cudaMalloc((float**)&device_array, sizeof(float) * N);

    // 2. memory copy host to device
    cudaMemcpy(device_array, host_array, sizeof(float) * N, cudaMemcpyHostToDevice);

    // 3. call kernel function
    int blockSize = 32;
    dim3 block(blockSize, 1, 1);
    dim3 grid((N + blockSize - 1) / blockSize, 1, 1);
    plusOneArray<<<grid, block>>>(device_array);
    // parallelize -> for (auto &e : host_array) e+= 1.0f;

    // 4. memory copy device to host
    cudaMemcpy(host_array, device_array, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
        printf("host_array[%d]: %f\n", i, host_array[i]);

    // 5. free device memory
    cudaFree(device_array);
}
```

どうでしょうか。途中見慣れない表記もあったかと思いますが、main関数だけに関して言えば基本的にはC言語と同じく、mallocで領域を確保して、計算して、freeによってメモリを解放する・・・といった普遍的な流れですね。  

### 3.1. メモリ確保と解放
GPUのメモリを確保するために**デバイスで利用するためのポインタ**をホスト側で宣言します。(ここでは```int* device array```)  ホストとデバイス間でやり取りがしたいデータがある場合には必ずこの前段階で宣言する必要があります。  
ポインタを宣言したら、```cudaMalloc(void** ptr, size_t nbytes)``` により **デバイスメモリ**に領域を割り当てます。ここから先はもう```cudaMalloc```により割り与えられた領域は**ホスト側から参照することは出来なく**なります。使い終えたら、```cudaFree(void* ptr)```によってデバイスメモリを解放してあげましょう。

### 3.2. データ転送
2, 4ではホストメモリ、デバイスメモリ間のデータのコピーを行います。```cudaMemcpy(void *dst, void *src, size_t nbytes enum cudaMemcpyKind)```によってメモリコピーを行います。cudaMemcpyKindに関してはHost から Deviceへのコピーならば```cudaMemcpyHostToDevice```を選べば良いです。初期化に関してはcudaMemset等もあるので調べてみてください。

### 3.3. 並列処理
### 3.3.1. カーネル関数
並列処理を行う関数を見ていきます。GPUデバイス上で実行されるコードのことを**カーネル**と呼びます。今回では、```__global__ void plusOneArray()```で示される関数に当たります。カーネル関数を定義するときには、`__global__`という修飾子が必要となります。また、`__global__`関数内で呼び出す関数を定義するためには`__device__`修飾子を宣言して関数を定義する必要があります。 `__device__`関数はデバイスからの呼び出し限定であり、ホストからは呼び出せません。また、カーネル関数の返り値には**void**型しか許されていないことに注意してください。
ホストからカーネル関数を呼び出すとデバイスでの実行に移ります。デバイス内では大量のスレッドが生成され各スレッドで処理が実行されます。   
ちなみに、カーネルが呼び出されるとすぐに制御がホストに戻るため、カーネルがGPUで実行されている間にホスト側では他の関数を実行することができます。 また、カーネル関数に渡した引数は自動的にメモリが確保されますので、`int`や`float`などの定数はそのまま引き渡すことができます。  
注意点として、c++でよく用いられているSTL(Standard Template Library)などをカーネルに渡しても正常に動作はしないので（コンパイルできない？）、cベースで記述する必要があります。

### 3.3.2. スレッドの構成
CUDAでは**スレッド**は階層状に抽象化されています。スレッド階層は複数のスレッドが集まった**ブロック**と、ブロックが複数集まった**グリッド**の2層構造からなります。グリッド内のスレッド間では、全て同じグローバルメモリ空間を共有します。  
このスレッド数とブロック数を決定することで**並列化のサイズ**が決定されます。そのため、特定の並列化サイズに基づいてグリッドとブロックのサイズを決定する必要があります。

**スレッドの構成**（[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)より）
![cuda-grid](grid-of-thread-blocks.png)  

### 3.3.3. グリッドの決定
並列数を決定づけるスレッド数はグリッドサイズとブロックサイズをホストで定義することにより決まります。CUDAでは、グリッドとブロックは3次元で構成されます。これにより、画像や体積などの領域内の要素にまたがる計算を自然に行うことができます。サイズは、```dim3 block(sizeX, sizeY, sizeZ)```のように決定されます。基本は```int blockSize = 32```のようにブロックサイズを先に決めて、並列化したいサイズNから、  
```c
dim3 block(blockSize, 1, 1);
dim3 grid((N + blockSize - 1) / blockSize, 1, 1);
```
のように並列数を決定をします。この場合、```blockSize * (N * blockSize - 1) / blockSize```分のスレッドが実行されることになります。

### 3.3.5. スレッドのインデックス
所望の処理をカーネル関数に書いていくことになると思いますが、カーネル内部では、どのようにスレッドの番地を知ることができるのでしょうか。`__global__`関数内には、スレッドは互いを区別するために以下の組み込み変数が用意されています。
```c
blockDim.x
blockIdx.x
threadIdx.x
```
`blockDim`は**ブロックサイズ**、`blockIdx`は**ブロックのインデックス**、`threadIdx`は**ブロック内におけるスレッドのインデックス**を示しています。そのため、グリッド内のスレッドのインデックスは次のように表せます。
```c
unsigned int u = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int v = blockDim.y * blockIdx.y + threadIdx.y;
unsigned int w = blockDim.z * blockIdx.z + threadIdx.z;
```
このようにグリッド内のスレッドを**一意に識別する**ことができるので、スレッドとデータ要素とのマッピングが可能になります。
今回のコードではスレッドの番地を用いて配列の番地を指定して、要素に+1しています。 

なお、並列数がブロックサイズで割り切れず、スレッド数が配列の要素数を超えてしまう場合があります。そのため、次のような要素数を超えたインデックスに関しては処理を行わないといった記述が行われます。
```c
__global__ void function(float* matrix, int row, int col) {
    unsigned int u = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int v = blockDim.y * blockIdx.y + threadIdx.y;
    if (u > row || v > col) return; // over matrix array length

    matrixCalc(matrix, u, v);
}
```

### 3.3.6. カーネルの起動
準備は整ったのでカーネルを起動しましょう。ホストから、`function<<<grid, block>>>()`のように実行します。
このように実行すれば、事前に設定しておいたグリッドサイズとブロックサイズに従ってスレッドが起動し、処理が行われます。  
内部の動作としてはカーネルの実行はブロック単位ごとに行われていきます。今回のコードだと、32スレッドごとの処理が32回行われることになりますね。  
もう少し厳密に話すと、スレッドが32個ずつ、ワープ(warp)と呼ばれるグループにまとめられた上で実行されます。ワープスケジューラによって32個のスレッド分のワープに分割され、利用可能なハードウェアのリソースに割り当てられていきます。

### 3.4. コードの検証
さて、動作を確認していきましょう。前述したように`nvcc`コマンドを用いて`nvcc main.cu`でコンパイルすると、実行ファイル`./a.out`が生成されますので、実行します。
カーネル関数内では`printf`が使えますので今回は`printf`でスレッドの番地と実際に配列の要素に対して+1されているかどうかを確認した結果です。for文なしに全ての配列にアクセスして計算できています。特徴的なのは、ブロックごとにカーネルが走っているのがなんとなく分かるところでしょうか。
![cuda-res](result.png) 

そしてホストメモリにコピーした後の結果も見てみましょう。

ホスト側で計算はしていませんが、ちゃんと`host_array`の値が更新されたことを確認できました。
![cuda-res2](result2.png) 

## まとめ
というわけでざっくりとCUDAについて見てきました。（公式のCUDA Cプログラミングガイドの100分の1くらいです）  
どう並列化させるかがミソだと思うので、色々試してみてください。  
次はCT画像再構成を具体例として、プログラムを見ていきたいと思います。


### 以下Tips
#### T.1. クラスをglobal関数に渡した後に内部でメンバ関数を実行できるか
#### T.2. Shared Memoryを使って高速化
#### T.3. cudaMallocManagedでホストとデバイスから参照しよう
#### T.4. mallocとfreeがめんどくさい時は
#### T.5. async allocation
#### T.6. マジックナンバー 32
#### T.7. thrustを使ってコードをきれいに

## 参考資料  
一次情報 

CUDA全般について（[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)）  
CUDA Programmingの全て（[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)）   
CUDA 環境構築 （[CUDA quick start guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/)）  
公式サンプル （[cuda-samples](https://github.com/NVIDIA/cuda-samples)）

二次情報

CUDA全体についてわかりやすくまとまっているサイト [See This](https://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?CUDA)

# CT画像再構成
## X線CTとは
**X線CT（X-ray Computed Tomography）** は、我々の研究室でのコンテクストでは、産業分野における非破壊検査技術として扱われており、製品や材料の内部構造を詳細に可視化するために使用されます。（画像は超大型CT装置、[fraunhofer IIS](https://www.iis.fraunhofer.de/en/ff/zfp/tech/hochenergie-computertomographie.html)より引用）  
<div align="center">
<img src="fraunhofer.jpeg" width="60%"> 
</div>
  

X線CTは、X線と検出器を組み合わせて使用します。被検査物にX線を照射し、その後、検出器がX線の**透過率**を測定します。これにより、被検査物の内部でのX線の**線減弱係数**（物質の密度に比例）や散乱強度のパターンを得ることができます。

CT再構成のプロセスについて説明します。まず、複数の角度からのX線撮影を行います。非検査物がX線源と検出器の間を回転しながら、被検査物を一定の回転間隔でスキャンします。このプロセスにより、被検査物周囲の透過率の断層画像が取得されます。取得された断層画像から再構成計算を行うことにより、最終的な三次元の内部構造を表現することができます。  

再構成の方法には、**フィルター逆投影法（FBP: Filtered Back Projection）** と **逐次近似再構成法（IR: Iterative Reconstruction）** に大別されます。今回の実装では逐次近似再構成法について実装していきたいと思います。  

## フィルター逆投影法
今回はフィルター逆投影法については実装しませんが、簡単に説明だけします。フィルター逆投影法は、得られた投影像に対して特定の周波数フィルターを掛けた後に、投影線上に投影値を加算していく手法になります。  
  
理論を学ぶ場合はこちら（[Radon変換](http://racco.mikeneko.jp/Kougi/2012s/IPPR/2012s_ippr12.pdf)と[フィルター逆投影](http://racco.mikeneko.jp/Kougi/2012s/IPPR/2012s_ippr13.pdf)）がよくまとまっていて分かりやすいと思います。   
以下概略  
投影像分布を $$g(s, \theta)$$ 、再構成領域における物体の分布を $$f(x, y)$$ としたとき、 $f(x, y)$ を2次元フーリエ変換したときに現れる関数 $F(u=\xi cos\theta, v=\xi sin\theta)$ が、投影像の1次元フーリエ変換 $G(\xi, \theta)$ と合致すること（投影定理）を利用することにより導出される。極座標変換時に現れる係数であるRampフィルタ $\lvert\xi\rvert,(-\infty<\xi<\infty)$ が理想的なフィルタである。が、有限の領域でカットする必要があるので、特定のフィルタを選択する必要がある。    

**投影原理**
![radon](radon.png)

## 逐次再構成法

