// 一种求和的规约方法
// 利用原子操作
// 1.将数据分块，每个块内的数据进行规约
// 2.将每个块内的规约结果进行规约

#include<stdio.h>
#include<math.h>

#define N 1e+8
#define BLOCK_SIZE 256
#define GRID_SIZE 32

//定义规约的数据
__managed__ int source[(int)N];

// 可拓展性
__managed__ int gpu_result[1] = {0};
__managed__ int cpu_result[1] = {0};



//
// @brief: 一种求和的规约方法
// @param: in 输入数据
// @param: count 输入数据的个数
// @param: out 输出数据
// @return: void
//
__global__ void sum_gpu(int *in,int count,int *out){
    __shared__ int cache[BLOCK_SIZE];

    // a00 a01 | a02 a03 
    // a10 a11 | a12 a13
    // -----------------
    // a20 a21 | a22 a23
    // a30 a31 | a32 a33


    int shared_tmp = 0;
    //首先读取a00、a02、a20、a22相加
    //其余同理
    for(int idx = blockIdx.x*blockDim.x+threadIdx.x;idx < count;idx += blockDim.x*gridDim.x){
        shared_tmp += in[idx];
    }
    //将结果存储到共享内存中
    cache[threadIdx.x] = shared_tmp;
    __syncthreads();

    //一个块内的求和规约
    int tmp = 0;
    for(int step = blockDim.x/2;step > 0;step /= 2){
        if(threadIdx.x < step){
            tmp = cache[threadIdx.x] + cache[threadIdx.x+step];
        }
        __syncthreads();
        cache[threadIdx.x] = tmp;
    }

    if(blockIdx.x*blockDim.x<count){
        if(threadIdx.x == 0){
            //原子操作
            atomicAdd(out,cache[0]);
        }
    }
}

//
// @brief: cpu求和
// @param: in 输入数据
// @param: count 输入数据的个数
// @param: out 输出数据
// @return: void
//
void sum_cpu(int *in,int count,int *out){
    int sum = 0;
    for(int i = 0;i < count;i++){
        sum += in[i];
    }
    out[0] = sum;
}

int main(void){
    //初始化
    printf("Init input data...\n");
    for(int i = 0;i < N;i++){
        source[i] = rand()%10;
    }

    cudaEvent_t start,stop_gpu,stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start,0);
    cudaEventSynchronize(start);
    for(int i = 0;i<20;i++){
        gpu_result[0] = 0;
        sum_gpu<<<GRID_SIZE,BLOCK_SIZE>>>(source,N,gpu_result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_gpu,0);
    cudaEventSynchronize(stop_gpu);

    sum_cpu(source,N,cpu_result);

    cudaEventRecord(stop_cpu,0);
    cudaEventSynchronize(stop_cpu);

    float cpu_time,gpu_time;
    cudaEventElapsedTime(&gpu_time,start,stop_gpu);
    printf("gpu Time to generate: %3.1f ms\n",gpu_time/20);
    cudaEventElapsedTime(&cpu_time,stop_gpu,stop_cpu);
    printf("cpu Time to generate: %3.1f ms\n",cpu_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(stop_cpu);

    bool error = false;
    for(int i = 0;i < 1;i++){
        if(gpu_result[i] != gpu_result[0]){
            error = true;
            printf("Error: gpu_result[%d] = %d\n",i,gpu_result[i]);
        }
    }
    printf("Result %s\n",error?"Failed":"Success");

    return 0;
}