//简单的矩阵乘法

#include <stdio.h>
#include <math.h>

//定义block的大小
#define BLOCK_SIZE 32

//
// @brief: cpu矩阵乘法
// @param: a, b, c, size
// @paramDesc: a, b, c分别是两个矩阵和结果矩阵的指针，size是矩阵的大小
// @return: void
//
void cpu_matrix_mult(int *a, int *b, int *c, const int size)
{
    for(int y=0; y<size; ++y)
    {
        for(int x=0; x<size; ++x)
        {
            int tmp = 0;
            //在第x行和第y列进行乘法运算
            //step是第x行和第y列的元素个数
            //作图帮助理解

            //                         b00 b01 b02 b03
            //                         b10 b11 b12 b13
            //                         b20 b21 b22 b23
            //                         b30 b31 b32 b33
            //
            // a00 a01 a02 a03         c00 c01 c02 c03
            // a10 a11 a12 a13         c10 c11 c12 c13    
            // a20 a21 a22 a23         c20 c21 c22 c23
            // a30 a31 a32 a33         c30 c31 c32 c33
            //
            //                          x=2  y=1
            // c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
            //          step0      step1       step2       step3

            for(int step = 0; step < size; ++step)
            {
                tmp += a[y*size + step] * b[step * size + x];
            }
            c[y * size + x] = tmp;
        }
    }
}


//
// @brief: gpu矩阵乘法
// @param: a, b, c, size
// @paramDesc: a, b, c分别是两个矩阵和结果矩阵的指针，size是矩阵的大小
// @return: void
//
__global__ void gpu_matrix_mult(int *a, int *b, int *c, const int size)
{
    //计算当前线程的全局坐标
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    int tmp = 0;

    //判断当前线程是否在矩阵范围内
    //容易遗忘的一步
    if( x < size && y < size)
    {
        for( int step = 0; step < size; ++step)
        {
            tmp += a[y * size + step] * b[step * size + x];
        }
        c[y * size + x] = tmp;
    }
}



int main()
{
    //定义矩阵的大小
    int matrix_size = 1000;

    //计算矩阵的内存大小
    int memsize = sizeof(int) * matrix_size * matrix_size;

    //分配内存
    int *h_a, *h_b, *h_c, *h_cc;

    //cudaMallocHost分配的内存是固定内存，可以被cpu和gpu访问
    //谨慎使用固定内存，因为它受限于系统上可用的物理RAM。分配大量的固定内存可能导致内存耗尽问题。
    //使用cudaFreeHost释放内存
    cudaMallocHost( (void**)&h_a, memsize);
    cudaMallocHost( (void**)&h_b, memsize);
    cudaMallocHost( (void**)&h_c, memsize);
    cudaMallocHost( (void**)&h_cc, memsize);

    //初始化矩阵
    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            h_a[y * matrix_size + x] = rand() % 1024;
        }
    }

    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            h_b[y * matrix_size + x] = rand() % 1024;
        }
    }

    //分配GPU内存
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a , memsize);
    cudaMalloc((void**) &d_b , memsize);
    cudaMalloc((void**) &d_c , memsize);

    //将数据从cpu内存拷贝到gpu内存
    cudaMemcpy( d_a, h_a, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, memsize, cudaMemcpyHostToDevice);

    //定义grid和block的大小
    unsigned int grid_rows = (matrix_size +BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_cols = (matrix_size +BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);//1.gpu warp 32 2. <= 1024

    //调用kernel函数
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, matrix_size);

    //将结果从gpu内存拷贝到cpu内存
    cudaMemcpy( h_c, d_c, memsize, cudaMemcpyDeviceToHost);

    //调用cpu函数
    cpu_matrix_mult(h_a, h_b, h_cc, matrix_size);

    //验证结果
    bool errors = false;
    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            if(fabs(h_cc[y*matrix_size + x] - h_c[y*matrix_size + x]) > (1.0e-10))
            {
                //printf("%d, %d\n", y, x);
                errors = true;
            }
        }
    }
    printf("Result: %s\n", errors?"Errors":"Passed");

    //释放内存
    cudaFreeHost(h_a );
    cudaFreeHost(h_b );
    cudaFreeHost(h_c );
    cudaFreeHost(h_cc );
    cudaFree(d_a );
    cudaFree(d_b );
    cudaFree(d_c );
    return 0;

}


