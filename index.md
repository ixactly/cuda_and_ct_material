# 大竹研 講習会
## CUDAとは
**CUDA**(Compute Unified Device Architecture)はNVIDIA社によるGPUプログラミング開発環境です。通常、一般的なコンピュータはCPUを使用して計算を行いますが、 CUDAを使用することでGPUも計算に利用することができます。
GPUは、多数のコアと高い並列処理能力により、 **並列化が可能な演算**を高速に処理することができます。（単一の処理自体は速くないことに注意）
CUDAは、CやC++をベースとしたプログラミングモデルを提供しており、GPUを使用した並列計算を容易に実装することができます。GPUの性能を真に引き出して高速化したい場合にはハードウェアアーキテクチャに対する理解が必要になるため、難しいです（自分もあまり分かっていません）。逆に言うとハードウェアを意識したプログラミングも可能であると言うことです。  

## CUDAプログラミング
CUDAで出来ることは、単一の命令を与えて、複数のデータを得ることです。（Single Program(Instruction) Multiple Data, SPMD, SIMD）  
そこでCUDAでは、C，C++で書かれた関数に対して並列化をかけることによって高速化を目指します。ここでは、並列化するための準備を行なっていきます。
### 0. CUDAとCの相違点
CUDAは基本的にはC言語とほとんど相違はありません。そのため、Cで書かれたプログラムに関してはそのまま動作すると思われます。
CUDAではCに対して大きく2つの拡張がなされています。
1. ホストメモリ（CPU）とデバイスメモリ（GPU）間でデータを転送する機能
2. ホスト（CPU）から、デバイス（GPU）に対して、多数のスレッドを並列に命令を与える機能  
   
CUDAのパラダイムでは、CPU側を**ホスト**、GPU側を**デバイス**と呼びます。1について、ホストとデバイスは異なるメモリシステムとなっているため、ホスト側からデバイスメモリを参照することは出来ません（逆も然り。厳密には異なるかもしれないが、大まかにはRAMがホストメモリ、VRAMがデバイスメモリと思ってもらって良い）。そのため、並列計算に必要となるボリュームデータや点群データなどの入力データはデバイスメモリに事前に転送して置く必要があります。また、計算後の出力データもデバイスからホストへと転送しなければなりません。2についてはそのままの意味で、並列計算の命令を与えます。基本的にはforループなどの、indexが変わるけれど、実行内容が変わらないような命令を並列化することになるでしょう。

### 1. プログラミングの流れ
一般的なCUDAプログラムの構造は主に以下の5つのstepに分かれます。
1. GPUメモリの確保
2. CPUメモリからデータをGPUメモリにコピー
3. CUDAカーネルを呼び出し、並列計算を実行する
4. GPUメモリからデータをCPUメモリにコピー
5. GPUメモリの解放

実際に書くと以下のようなプログラムになるでしょう。ここでは与えた配列に対して全要素に1を足すというプログラムを例として示します。

```c++
__global__ void plusOneArray(int* array) {
    unsigned int u = block.Idx + 
    array[u] += 1.0f;
}

int main() {
    int N = 5;
    // init host memory
    float host_array[5] = {3.f, 2.f, 5.f, 1.f, 4.f};
    // init device memory
    float* device_array;

    // 1. allocate device memory
    cudaMalloc((float**)&device_array, sizeof(int) * N);

    // 2. memory copy host to device
    cudaMemcpy(device_array, host_array, sizeof(int) * N, cudaMemcpyHostToDevice);

    // 3. call kernel function
    plusOneArray<<<grid, block>>>(device_array);
    // parallelize -> for (auto &e : host_array) e+= 1.0f;

    // 4. memory copy device to host
    cudaMemcpy(host_array, device_array, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // 5. free device memory
    cudaFree(device_array);
}
```

どうでしょうか。main関数だけに関して言えば基本的にはC言語と同じく、mallocで領域を確保して、計算して、freeによってメモリを解放する・・・といった普遍的な流れですね。  

#### 1. メモリ確保と解放
GPUのメモリを確保するために**デバイスで利用するためのポインタ**をホスト側で宣言します。(ここでは```int* device array```)  ホストとデバイス間でやり取りがしたいデータがある場合には必ずこの前段階で宣言する必要があります。  
ポインタを宣言したら、```cudaMalloc(void** ptr, size_t nbytes)``` により **デバイスメモリ**に領域を割り当てます。ここから先はもう```cudaMalloc```により割り与えられた領域は**ホスト側から参照することは出来なく**なります。使い終えたら、```cudaFree(void* ptr)```によってデバイスメモリを解放してあげましょう。

#### 2. データ転送
2, 4ではホストメモリ、デバイスメモリ間のデータのコピーを行います。```cudaMemcpy(void *dst, void *src, size_t nbytes enum cudaMemcpyKind)```によってメモリコピーを行います。cudaMemcpyKindに関してはHost から Deviceへのコピーならば```cudaMemcpyHostToDevice```を選べば良いです。初期化に関してはcudaMemset等もあるので調べてみてください。

#### 3. カーネル関数

というわけでざっくりとCUDAについて見てきました。どう並列化させるかがミソだと思うので、色々試してみてください。

以下Tips

#### T.1. クラスをglobal関数に渡した後に内部でメンバ関数を実行できるか
#### T.2. Shared Memoryを使って高速化
#### T.3. cudaMallocManagedでホストとデバイスから参照しよう
#### T.4. mallocとfreeがめんどくさい時は
#### T.5. streamとasync allocation
#### T.6. マジックナンバー 32
#### T.7. thrustを使ってコードをきれいに