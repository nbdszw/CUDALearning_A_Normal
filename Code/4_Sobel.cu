// 简单的使用Sobel算子做卷积的加速
// 需要使用OpenCV库

#include<opencv2/opencv.hpp>
#include<iostream>
#include"error.cuh" //CHECK函数，寻找错误信息

using namespace std;
using namespace cv;

//
// @brief: GPU使用Sobel算子做卷积
// @param: in 输入图像
// @param: out 输出图像
// @param: Height 图像高度
// @param: Width 图像宽度
// @return: void
//
__global__ void sobel_gpu(unsigned char *in,unsigned char *out,const int Height,const in Width){
    //获取线程索引
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    //获取全局索引
    int index = y*width + x;

    int Gx = 0;
    int Gy = 0;

    unsigned char x0,x1,x2,x3,x4,x5,x6,x7,x8;

    //SM register
    if(x>0&&x<(Width-1)&&y>0&&y<(Height-1)){
        //获取周围像素值
        x0 = in[(y-1)*Width+(x-1)];
        x1 = in[(y-1)*Width+x];
        x2 = in[(y-1)*Width+(x+1)];
        x3 = in[y*Width+(x-1)];
        x4 = in[y*Width+x];
        x5 = in[y*Width+(x+1)];
        x6 = in[(y+1)*Width+(x-1)];
        x7 = in[(y+1)*Width+x];
        x8 = in[(y+1)*Width+(x+1)];

        //计算卷积
        Gx = -x0 + x2 - 2*x3 + 2*x5 - x6 + x8;
        Gy = -x0 - 2*x1 - x2 + x6 + 2*x7 + x8;

        //写入输出图像
        out[index] = (unsigned char)sqrt((double)(Gx*Gx+Gy*Gy));
    }
    //边缘像素的处理
    else{
        out[index] = 0;
    }
}

int main(void){
    //读取图像
    Mat img = imread("imori.jpg",0);
    int height = img.rows;
    int width = img.cols;

    //高斯滤波
    Mat gaussImg;
    GaussianBlur(img,gaussImg,Size(3,3),0,0,BORDER_DEFAULT);

    //定义GPU输入输出图像
    Mat dst_gpu(height,width,CV_8UC1,Scalar(0));

    //定义GPU输入输出图像大小
    int memsize = sizeof(unsigned char)*height*width;

    //分配GPU内存
    unsigned char *dev_in,*dev_out;
    cudaMalloc((void**)&dev_in,memsize);
    cudaMalloc((void**)&dev_out,memsize);

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid((width+threadsPerBlock.x-1)/threadsPerBlock.x,(height+threadsPerBlock.y-1)/threadsPerBlock.y);

    CHECK(cudaMemcpy(dev_in,gaussImg.data,memsize,cudaMemcpyHostToDevice));

    //GPU计算
    sobel_gpu<<<blocksPerGrid,threadsPerBlock>>>(dev_in,dev_out,height,width);

    //将结果拷贝回主机
    cudaMemcpy(dst_gpu.data,dev_out,memsize,cudaMemcpyDeviceToHost);

    //保存结果
    imwrite("sobel_gpu.jpg",dst_gpu);

    //释放内存
    cudaFree(dev_in);
    cudaFree(dev_out);

    return 0;

}

