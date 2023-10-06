// 利用共享内存加速矩阵转置
// 1. 读取矩阵到共享内存
// 2. 计算写的索引
// 3. 写入矩阵


#include<stdio.h>
#include<math.h>

// 定义矩阵大小和BLOCK
#define BLOCK_SIZE 16
#define M 3000
#define N 1000

__managed__ int matrix[N][M];
__managed__ int gpu_result[M][N];
__managed__ int cpu_result[M][N];

//
// @brief:简单的GPU矩阵转置
// @param:in 输入矩阵
// @param:out 输出矩阵
//
__global__ void gpu_matrix_transpose(int in[N][M], int out[M][N]){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<M && y<N){
        out[x][y] = in[y][x];
    }
}

//
// @brief:利用共享内存加速矩阵转置
// @param:in 输入矩阵
// @param:out 输出矩阵
//
__global__ void gpu_shared_matrix_transpose(int in[N][M], int out[M][N]){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //定义共享内存,多申请一行一列
    //填充解决访存冲突
    //用于存储矩阵
    __shared__ int tile[BLOCK_SIZE+1][BLOCK_SIZE+1];

    if(x<M && y<N){
        //读取
        tile[threadIdx.y][threadIdx.x] = in[y][x];
    }
    __syncthreads();

    //计算写的索引
    //blockIdx.y*blockDim.y可以理解为每个block的起始y坐标
    //blockIdx.x*blockDim.x可以理解为每个block的起始x坐标
    int x1 = blockIdx.y * blockDim.y + threadIdx.x;
    int y1 = blockIdx.x * blockDim.x + threadIdx.y;

    //写入
    if(x1<N && y1<M){
        out[y1][x1] = tile[threadIdx.x][threadIdx.y];
    }

}

//
// @brief:CPU矩阵转置
// @param:in 输入矩阵
// @param:out 输出矩阵
//
void cpu_matrix_transpose(int in[N][M], int out[M][N]){
    for(int y = 0;y<N;y++){
        for(int x = 0;x<M;x++){
            out[x][y] = in[y][x];
        }
    }
}

int main(void){
    // 初始化
    for(int y = 0;y<N;y++){
        for(int x = 0;x<M;x++){
            matrix[y][x] = rand()%1024;
        }
    }

    // 定义时间
    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((M+block.x-1)/block.x,(N+block.y-1)/block.y);

    // GPU计算
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    gpu_shared_matrix_transpose<<<grid,block>>>(matrix,gpu_result);
    cudaDeviceSynchronize();

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    
    // CPU计算
    cpu_matrix_transpose(matrix,cpu_result);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float gpu_time;
    float cpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop_gpu);
    cudaEventElapsedTime(&cpu_time, stop_gpu, stop_cpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(stop_cpu);

    printf("GPU time: %f ms\n", gpu_time);
    printf("CPU time: %f ms\n", cpu_time);

    bool error = false;
    for(int y = 0;y<N;y++){
        for(int x = 0;x<M;x++){
            if(gpu_result[y][x] != cpu_result[y][x]){
                error = true;
                break;
            }
        }
    }

    printf("Result: %s\n", error ? "FAIL" : "PASS");
    return 0;

}