// 对向量进行简单的操作,比较CPU与GPU的运行速度

#include <stdio.h>
#include <math.h>


//
// @brief: GPU核函数对向量进行加法操作
// @param: x, y, z, count
// @paramDesc: x, y, z分别为两个向量和结果向量, count为向量的长度
// @return: void
//
__global__ void vecAdd(const double *x, const double *y, double *z, int count)
{
    // 获取当前线程的全局索引
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // t00 t01 t02 t10 t11 t12 t20 t21 t22
    if( index < count)
    {
        z[index] = x[index] + y[index];
    }
}

//
// @brief: CPU函数对向量进行加法操作
// @param: x, y, z, count
// @paramDesc: x, y, z分别为两个向量和结果向量, count为向量的长度
// @return: void
//
void vecAdd_cpu(const double *x, const double *y, double *z, int count)
{
    for(int i = 0; i<count; ++i)
    {
        z[i] = x[i] + y[i];
    }
}


int main()
{
    // 定义向量长度
    const int N = 1000;

    // 计算向量占用内存大小
    const int MemSize = sizeof(double) * N;

    //在CPU上分配内存
    double *h_x = (double*) malloc(MemSize);
    double *h_y = (double*) malloc(MemSize);
    double *h_z = (double*) malloc(MemSize);
    double *result_cpu = (double*) malloc(MemSize);


    //初始化向量
    for( int i = 0; i<N; ++i)
    {
        h_x[i] = 1;
        h_y[i] = 2;
    }

    //在GPU上分配内存cudaMalloc
    //cudaMalloc分配全局内存,返回指向分配内存的指针
    double *d_x, *d_y, *d_z;
    cudaMalloc((void**) &d_x, MemSize );
    cudaMalloc((void**) &d_y, MemSize );
    cudaMalloc((void**) &d_z, MemSize );

    //将数据从CPU内存复制到GPU内存
    cudaMemcpy(d_x ,h_x ,MemSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_y ,h_y ,MemSize , cudaMemcpyHostToDevice);

    //定义线程块block大小和线程块数量grid大小
    const int block_size = 128;
    const int grid_size  = (N + block_size -1)/block_size;

    //调用核函数
    vecAdd<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    //将数据从GPU内存复制到CPU内存
    //隐同步
    cudaMemcpy( h_z, d_z, MemSize, cudaMemcpyDeviceToHost);

    //调用CPU函数
    vecAdd_cpu(h_x, h_y, result_cpu, N);

    //比较结果
    bool error = false;

    for(int i=0; i<N; ++i)
    {
        if(fabs(result_cpu[i] - h_z[i]) > (1.0e-10))
        {
            error = true;
        }
    }
    printf("Result: %s\n", error?"Errors":"Pass");

    //释放内存
    free(h_x);
    free(h_y);
    free(h_z);
    free(result_cpu);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

}


