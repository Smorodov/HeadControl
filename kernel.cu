#include <iostream>

#include <vector>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/gpu/devmem2d.hpp> 
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define PI 3.1415926535897932f

// these exist on the GPU side
texture<float4,2>  texImage;

#define BLCKDIM_X 8
#define BLCKDIM_Y 8

//-----------------------------------------------------------
//
//-----------------------------------------------------------
__global__ void ToTexture(DevMem2Df src,float* frame_dev) 
{        
	// Координаты текущего пикселя
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;
	// Ширина и высота входного изображения
	const int imageW=src.cols;
	const int imageH=src.rows;
	// Проверяем на изображении ли мы
	if (ix < imageW && iy < imageH)
	{
	frame_dev[iy*4*imageW+ix*4]=src.ptr(iy)[ix*3];
	frame_dev[iy*4*imageW+ix*4+1]=src.ptr(iy)[ix*3+1];
	frame_dev[iy*4*imageW+ix*4+2]=src.ptr(iy)[ix*3+2];
	frame_dev[iy*4*imageW+ix*4+3]=0;
	}
}
//-----------------------------------------------------------
//
//-----------------------------------------------------------
float* MatToTexture(const DevMem2Df src,texture<float4,2> &texSrc,int BLOCKDIM_X=8,int BLOCKDIM_Y=8)
{
	float* frame_dev=0;
	// Настройка конфигурации сетки
	dim3 block(BLOCKDIM_X,BLOCKDIM_Y);
	dim3 grid(ceil((float)src.step/(float)block.x), ceil(((float)src.rows)/(float)block.y));
	cudaSetDevice(0);
	//-----------------------------------------------------------
	// 
	//-----------------------------------------------------------
	int size=src.cols*src.rows;
	cudaMalloc((void**)&frame_dev, size * sizeof(float4));
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaBindTexture2D( 0, texSrc, frame_dev, desc, src.cols, src.rows, src.cols*sizeof(float4) );
	ToTexture<<<grid, block>>>(src,frame_dev);
	cudaDeviceSynchronize();		
	return frame_dev;
}

//-----------------------------------------------------------
// Шаблон для новых функций
//-----------------------------------------------------------
__global__ void kernel(DevMem2Df src, PtrStepf dst) 
{        
	// Координаты текущего пикселя
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;
	// Ширина и высота входного изображения
	const int imageW=src.cols;
	const int imageH=src.rows;
	// Проверяем на изображении ли мы
	if (ix < imageW && iy < imageH)
	{
        float4 clr = tex2D(texImage, ix, iy);
	
	// <<-----------------Сюда писать код

		dst.ptr(iy)[ix*3]=clr.x;
		dst.ptr(iy)[ix*3+1]=clr.y;
		dst.ptr(iy)[ix*3+2]=clr.z;
	}
}

//-----------------------------------------------------------
//
//-----------------------------------------------------------
cudaError_t VoterKernelHelper(const DevMem2Df src, PtrStepf dst)
{
	// Всегда Succsess :)
	cudaError_t cudaStatus=cudaSuccess;
	// ---------------------
	dim3 block(BLCKDIM_X,BLCKDIM_Y);
	// ---------------------
	dim3 grid(ceil((float)src.step/(float)block.x), ceil(((float)src.rows)/(float)block.y));
	// Не забыть освободить память на девайсе
	float *frame_dev=0;
	frame_dev=MatToTexture(src,texImage);
	// Вызов ядерной функции
	kernel<<<grid, block>>>(src, dst);
	// Синхронизация потоков на девайсе (х.з. надо или нет)
	cudaDeviceSynchronize();
	// Отцепим текстуру
	cudaUnbindTexture( texImage );
	// Освободили память на девайсе
	cudaFree(frame_dev);
	return cudaStatus;
}
