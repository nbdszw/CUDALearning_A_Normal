// CUDA中的Hello World
// 1. 编写kernel函数
// 2. 调用kernel函数
// 3. 编译运行 nvcc 1_HelloWorld.cu -o 1_HelloWorld
// 4. 查看结果 ./1_hello_world
// 5. 了解kernel函数的执行模式
// 6. 性能优化：通过nsys nvprof nvvp等工具进行分析

#include <stdio.h>

__global__ void my_first_kernel()
{   
    // 获取线程索引
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    // 打印线程信息，每个线程Hello World
    printf("Hello World from thread(thread index:(%d,%d), block index:(%d,%d))!\n", tidy, tidx, bidy, bidx);
}


// thread  --> block --> grid
// SM stream multi-processor
// total threads: block_size * grid_size
// block_size: num of threads in a block
// grid_size: num of blocks in a grid
int main()
{
    printf("Hello World from CPU!\n");

    dim3 block_size(3,3);
    //t00, t01, t02
    //t10, t11, t12
    //t20, t21, t22
    dim3 grid_size(2,2);
    //b00, b01
    //b10, b11

    my_first_kernel<<<grid_size,block_size>>>();

    // 容易忘记的一步：设备同步
    // 保证kernel函数执行完毕
    cudaDeviceSynchronize();

    return 0;
}
