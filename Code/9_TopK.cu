//选出数组中最大的k个数
//所用思考类似于8_ReduceAndAtomic

#include<stdio.h>
#include<math.h>
#include<time.h>

#define N 100000000 //数组大小
#define BLOCK_SIZE 256
#define GRID_SIZE 64
#define topk 20 //选出最大的20个数

__managed__ int source[int(N)];
__managed__ int gpu_result[topk];
__managed__ int _1_pass_result[topk*GRID_SIZE];

//
// @brief: 插入比数组中元素值大的
// @param: array: 数组
// @param: value: 要插入的值
// @param: length: 数组长度
//
__device__ __host__ void insert_value(int* array,int value,int length){
    //对处于数组不同位置的情况进行处理
    for(int i =0;i<length;i++){
        //相等不插入
        if(array[i] == value){
            return;
        }

        //小于数组最后一个数，不插入
        if(value<array[length-1]){
            return;
        }

        //否则，插入
        //插入位置的元素后移
        for(int i = length - 2;i>=0;i--){
            // 找到插入位置
            if(value>array[i]){
                array[i+1] = array[i];
            }

            //插入到最后一个数
            else{
                array[i+1] = value;
                return;
            }
        }
    }

    array[0] = value;
}

//
// @brief: 选出数组中最大的k个数
// @param: in: 输入数组
// @param: out: 输出数组
// @param: length: 数组长度
// @param: k: 选出最大的k个数
//
__global__ void gpu_topk(int* in,int * out,int length,int k){
    //每个线程块的共享内存
    __shared__ int sdata[BLOCK_SIZE * topk];

    //每个线程块的最大值数组
    int top_array[topk];
    
    //初始化数组
    for(int i =0;i<topk;i++){
        top_array[i] = INT_MIN;
    }

    //开始时每个线程处理每个数据块中的一个元素
    //与8中的思想相同，每个线程处理一个数据块中的一个元素
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x;idx<length;idx+=blockDim.x*gridDim.x){
        int value = in[idx];
        insert_value(top_array,value,topk);
    }

    //将每个线程块的最大值数组复制到共享内存中
    for(int i =0;i<topk;i++){
        sdata[threadIdx.x*topk+i] = top_array[i];
    }

    __syncthreads();

    //每个线程块内部进行归约
    for(int i = BLOCK_SIZE/2;i>=1;i/=2){
        if(threadIdx.x<i){
            for(int j = 0;j<topk;j++){
                //(threadIdx.x+i)*topk为每个线程块中的第i个线程的最大值数组的起始位置
                insert_value(top_array,sdata[(threadIdx.x+i)*topk+j],topk);
            }
        }
        __syncthreads();

        if(threadIdx.x<i){
            for(int j = 0;j<topk;j++){
                sdata[threadIdx.x*topk+j] = top_array[j];
            }
        }

        __syncthreads();
    }

    if(blockIdx.x*blockDim.x<length){
        if(threadIdx.x == 0){
            for(int i =0;i<topk;i++){
                out[blockIdx.x*topk+i] = top_array[i];
            }
        }
    }
}

//
// @brief: CPU版本的topk
// @param: in: 输入数组
// @param: out: 输出数组
// @param: length: 数组长度
// @param: k: 选出最大的k个数
//
void cpu_topk(int* in,int * out,int length,int k){
    for(int i = 0;i<length;i++){
        insert_value(out,in[i],k);
    }
}

int main(void){
    srand((unsigned)time(NULL));
    // 初始化数据
    printf("Init source data...........\n");
    for(int i=0;i<N;i++){
        source[i]=rand()%10000;
    }
    printf("Complete init source data.....\n");

    cudaEvent_t start,stop_gpu,stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    printf("GPU Run **************\n");
    for(int i = 0;i<20;i++){
        gpu_topk<<<GRID_SIZE,BLOCK_SIZE>>>(source,_1_pass_result,N,topk);

        gpu_topk<<<1,BLOCK_SIZE>>>(_1_pass_result,gpu_result,topk*GRID_SIZE,topk);
        cudaDeviceSynchronize();
    }
    printf("GPU Complete!!!\n");

    cudaEventRecord(stop_gpu,0);
    cudaEventSynchronize(stop_gpu);
    
    int cpu_result[topk] = {0};

    printf("CPU RUN***************\n");
    cpu_topk(source,cpu_result,N,topk);


    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    printf("CPU Complete!!!!!");

    float gpu_time,cpu_time;
    cudaEventElapsedTime(&gpu_time,start,stop_gpu);
    cudaEventElapsedTime(&cpu_time,stop_gpu,stop_cpu);

    printf("GPU time: %f ms\n",gpu_time/20);
    printf("CPU time: %f ms\n",cpu_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(stop_cpu);

    bool error = false;
    for(int i=0;i<topk;i++){
        printf("CPU top%d: %d; GPU top%d: %d;\n", i+1, cpu_result[i], i+1, gpu_result[i]);
        if(cpu_result[i]!=gpu_result[i]){
            error = true;
            break;
        }
    }

    printf("Test %s\n",error?"Failed":"Successed");
    
}