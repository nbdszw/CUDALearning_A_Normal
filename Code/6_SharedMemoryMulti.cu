//
// 利用共享内存和统一内存优化矩阵乘法
// 优化思路：将矩阵分块，每个线程块负责一个子矩阵的计算
// 复习：线性代数的分块矩阵乘法
// 步骤：
// 1.将矩阵分块
// 2.将子矩阵加载到共享内存
// 3.计算子矩阵乘法
// 4.将结果写回全局内存
//

// a[][] * b[][] = c[][]
// 
//                         b00 b01 b02 b03
//                         b10 b11 b12 b13
//                         b20 b21 b22 b23
//                         b30 b31 b32 b33
//
// a00 a01 a02 a03         c00 c01 c02 c03
// a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
// a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33         c30 c31 c32 c33     c30 c31
//
//                              b00 b01->  sub_b_step_0
//                              b10 b11
//
//                              b20 b21->  sub_b_step_1
//                              b30 b31

// sub_a_step_0 sub_a_step_1    sub_c
// a20 a21      a22 a23         c20 c21
// a30 a31      a32 a33         c30 c31
//
// sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
//
// for(int step =0; step < N/block_size; step++ )
//      load sub_a_step to shared memory;
//      load sub_b_step to shared memory;
//      tmp += sub_a_step_on_sharedmemory * sub_b_step_on_sharedmemory;
// sub_c = tmp;
//
// cudaMalloc -> global memory
// data global memory -> shared memory
// threads shared memory -> register
// shared memory SM(stream multi-processor) same block same shared memory
//
// c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
// a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
// 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
// b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33

#include<stdio.h>
#include<math.h>

//定义矩阵的尺寸
#define M 1000
#define N 500
#define K 1000

//block
#define blocksize 16

//申请统一内存，定义矩阵
//统一内存：无需显式地复制数据。这有助于减少复杂的内存传输操作，提高编程的便捷性，并允许应用程序更好地利用GPU的计算能力。
__managed__ int a[M*N];
__managed__ int b[N*K];
__managed__ int c_gpu[M*K];
__managed__ int c_cpu[M*K];


//
// @brief: GPU矩阵乘法
// @param: a 输入矩阵
// @param: b 输入矩阵
// @param: c 输出矩阵
// @param: m 矩阵a的行数
// @param: n 矩阵a的列数，矩阵b的行数
// @param: k 矩阵b的列数
// @return: void
//
__global__ void gpu_matrix(int*a ,int *b,int *c,int m,int n,int k){
    //定义共享内存
    //大小为块的大小
    __shared__ int sub_a[blocksize][blocksize];
    __shared__ int sub_b[blocksize][blocksize];

    //获取索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tmp = 0;
    int idx;

    //load submatrixA
    for(int step = 0;step<=n/blocksize;step++){
        // 矩阵与线程对应
        // (step_x, step_y)为子矩阵对应线程的开始坐标
        // step*blocksize可理解为当前子矩阵的第一个线程的横坐标

        // a00 a01 | a02 a03         b00 b01 | b02 b03              thread(0,0)  thread(0,1)  thread(0,2)  thread(0,3)
        // a10 a11 | a12 a13         b10 b11 | b12 b13              thread(1,0)  thread(1,1)  thread(1,2)  thread(1,3)
        // -----------------         -----------------              thread(2,0)  thread(2,1)  thread(2,2)  thread(2,3)
        // a20 a21 | a22 a23         b20 b21 | b22 b23              thread(3,0)  thread(3,1)  thread(3,2)  thread(3,3)   
        // a30 a31 | a32 a33         b30 b31 | b32 b33    

        // 以blocksize=2为例，子矩阵的大小为2*2
        // step = 0时，子矩阵的坐标为(0,0) idx = 0
        // step = 1时，子矩阵的坐标为(0,2) idx = 2
        // step = 2时，子矩阵的坐标为(2,0) idx = 8
        // step = 3时，子矩阵的坐标为(2,2) idx = 10

        //循环执行过程：
        // step = 0时，子矩阵的坐标为(0,0) idx = 0 读入a00 a01 a10 a11
        // step = 1时，子矩阵的坐标为(0,2) idx = 2 读入a02 a03 a12 a13


        int step_x = step * blocksize + threadIdx.x;
        int step_y = y;
        idx = step_y * n + step_x;

        //不平均分块的情况：将超出边界的部分赋值0
        if(step_x >= n ||step_y>=m){
            sub_a[threadIdx.y][threadIdx.x] = 0;
        }
        else{
            // 否则，读入数据进入共享内存
            sub_a[threadIdx.y][threadIdx.x] = a[idx];
        }

        //load submatrixB
        //与上述同理
        step_x = x;
        step_y = step*blocksize + threadIdx.y;
        idx = step_y * k + step_x;

        //不平均分块的情况：将超出边界的部分赋值0
        if(step_x >= k ||step_y>=n){
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }
        else{
            sub_b[threadIdx.y][threadIdx.x] = b[idx];
        }

        //同步，等待所有线程加载完毕
        //最容易遗忘的一步
        __syncthreads();

        //计算子矩阵乘法
        for(int i =0;i<blocksize;i++){
            tmp += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if(x<k && y<m){
        c[y*k + x] = tmp;
    }
    
}

//
// @brief: CPU矩阵乘法
// @param: a 输入矩阵
// @param: b 输入矩阵
// @param: c 输出矩阵
// @param: m 矩阵a的行数
// @param: n 矩阵a的列数，矩阵b的行数
// @param: k 矩阵b的列数
// @return: void
//
void cpu_matrix(int*a ,int *b,int *c,int m,int n,int k){
    for(int y = 0;y<m;y++){
        for(int x = 0;x<k;x++){
            int tmp = 0;
            for(int step = 0;step<n;step++){
                tmp += a[y*n + step] * b[step*k + x];
            }
            c[y*k + x ] = tmp;
        }
    }
}


int main(){
    //初始化矩阵
    for(int y=0;y<M;y++){
        for(int x= 0;x<N;x++){
            a[y*N + x] = rand()%1024;
        }
    }

    for(int y=0;y<N;y++){
        for(int x=0;x<K;x++){
            b[y*K + x] = rand()%1024;
        }
    }

    unsigned int grid_x = (K+blocksize -1)/blocksize;
    unsigned int grid_y = (M + blocksize - 1)/blocksize;

    dim3 grid(grid_x,grid_y);
    dim3 block(blocksize,blocksize);

    
    gpu_matrix<<<grid,block>>>(a,b,c_gpu,M,N,K);
    cudaDeviceSynchronize();
    cpu_matrix(a,b,c_cpu,M,N,K);

    bool error = false;
    for(int y = 0;y<M;y++){
        for(int x= 0;x<K;x++){
            if(fabs(c_cpu[y*K+x] - c_gpu[y*K+x])>(1.0e-3)){
                error = true;
            }
        }
    }

   printf("Result: %s\n", error?"Error":"Pass");

    return 0;
}