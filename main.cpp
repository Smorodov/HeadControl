
#define  _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <direct.h>
#include "fstream"
#include "iostream"
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;


//cudaError_t VoterKernelHelper(const DevMem2Df src, PtrStepf dst);
cv::gpu::CascadeClassifier_GPU cascade_gpu;

//-------------------------------------------------------------------------------------------------------------
vector<Rect> detect_faces(Mat& image)
{
    vector<Rect> res;
	bool findLargestObject = true;
    bool filterRects = true;
	int detections_num;
	Mat faces_downloaded;
	Mat im(image.size(),CV_8UC1);
	GpuMat facesBuf_gpu;
	if(image.channels()==3)
	{
	cvtColor(image,im,CV_BGR2GRAY);
	}
	else
	{
	image.copyTo(im);
	}
	GpuMat gray_gpu(im);

    cascade_gpu.visualizeInPlace = false;
    cascade_gpu.findLargestObject = findLargestObject;
    detections_num = cascade_gpu.detectMultiScale(gray_gpu, facesBuf_gpu, 1.2,(filterRects || findLargestObject) ? 4 : 0,Size(image.cols/4,image.rows/4));
	

	if(detections_num==0){return res;}
	
	facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
	Rect *faceRects = faces_downloaded.ptr<Rect>();
	
	for(int i=0;i<detections_num;i++)
	{
		res.push_back(faceRects[i]);
	}
	
	return res;
}
//-----------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------
// Детектор контуров, вывод результатов
//----------------------------------------------------------------------
void DetectContour(Mat& img, Mat& image,Mat& Rad)
{
	Mat drawing( img.size(), CV_8UC3 );
	cvtColor(image,drawing,CV_GRAY2BGR);

	Mat img8U( img.size(), CV_8UC3 );
	cv::normalize(img,img,0,255,CV_MINMAX);
	img.convertTo(img8U,CV_8UC1);

	//Mat rect_12 = getStructuringElement(CV_SHAPE_RECT, Size(12,12) , Point(6,6));
	//erode(img8U, img8U, rect_12,Point(),1);
	//Mat rect_6 = getStructuringElement(CV_SHAPE_RECT, Size(6,6) , Point(3,3));
	//dilate(img8U,img8U,rect_6,Point(),2);


	vector<vector<Point> > contours;
	
	vector<Vec4i> hierarchy;

	findContours(img8U,contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point());

	if(contours.size()>0)
	{
		for( int i = 0; i < contours.size(); i++ )
		{
			if(contourArea(contours[i])>200)
			{				
				//drawContours( drawing, contours, i, Scalar(0,255,0), 1, 8, vector<Vec4i>(), 0, Point() );
				Rect r=cv::boundingRect(contours[i]);
				//cv::rectangle(drawing,r,Scalar(0,255,0));
				double m,M;
				Point P;
				//minMaxLoc(img(Rect(r)),&m,&M,0,&P);

				Moments mm;				
				mm=moments(img(Rect(r)));

				P.x=(mm.m10/mm.m00)+r.x;
				P.y=(mm.m01/mm.m00)+r.y;
				cv::circle(drawing,P,3,Scalar(0,255,0),1);
			}
		}
	}
	
	namedWindow( "Результат", CV_WINDOW_AUTOSIZE );
	imshow( "Результат", drawing );

}
//----------------------------------------------------------------------
// Расчет курватуры изофоты 
//----------------------------------------------------------------------
void IsofoteCurvatureDetector(Mat& img,Mat &Accumulator, Mat& Rad)
{
	// размер ядра фильтра (заметно влияет на детект)
	int gauss_kernel_size=9;
	// дисперсия для фильтра Гаусса
	double sigma=5;

	Mat Img32F(img.rows,img.cols,CV_32FC1);
	// Переводим изображение в серое одноканальное с элементами типа float
	img.convertTo(Img32F,CV_32FC1);

	GpuMat Img32F_d(Img32F);

	GpuMat Lx1(img.rows,img.cols,CV_32FC1);
	GpuMat Ly1(img.rows,img.cols,CV_32FC1);
	GpuMat Lx2(img.rows,img.cols,CV_32FC1);
	GpuMat Ly2(img.rows,img.cols,CV_32FC1);
	GpuMat Lxy(img.rows,img.cols,CV_32FC1);
	GpuMat Lx1sq(img.rows,img.cols,CV_32FC1);
	GpuMat Ly1sq(img.rows,img.cols,CV_32FC1);

	Mat Dx(img.rows,img.cols,CV_32FC1);
	Mat Dy(img.rows,img.cols,CV_32FC1);

	// Находим производные 
	Sobel(Img32F_d,Lx1,CV_32FC1,1,0,9);
	Sobel(Img32F_d,Ly1,CV_32FC1,0,1,9);
	Sobel(Img32F_d,Lx2,CV_32FC1,2,0,9);
	Sobel(Img32F_d,Ly2,CV_32FC1,0,2,9);
	Sobel(Img32F_d,Lxy,CV_32FC1,1,1,9);
	
	cv::gpu::sqr(Lx1,Lx1sq);
	cv::gpu::sqr(Ly1,Ly1sq);

	// Вынесем общую часть
	GpuMat K(img.rows,img.cols,CV_32FC1);

	GpuMat T1(img.rows,img.cols,CV_32FC1);
	GpuMat T2(img.rows,img.cols,CV_32FC1);
	GpuMat T3(img.rows,img.cols,CV_32FC1);
	GpuMat T4(img.rows,img.cols,CV_32FC1);

	cv::gpu::add(Lx1sq,Ly1sq,T1,cv::gpu::GpuMat(),CV_32FC1);
	cv::gpu::multiply(T1,-1,T1,1.0,CV_32FC1);

	cv::gpu::multiply(Ly1sq,Lx2,T2,1.0,CV_32FC1);

	cv::gpu::multiply(Lx1,2,T3,1.0,CV_32FC1);
	cv::gpu::multiply(T3,Lxy,T3,1.0,CV_32FC1);
	cv::gpu::multiply(T3,Ly1,T3,1.0,CV_32FC1);
	
	cv::gpu::multiply(Lx1sq,Ly2,T4,1.0,CV_32FC1);
	
	cv::gpu::add(T2,T3,T2,cv::gpu::GpuMat(),CV_32FC1);
	cv::gpu::add(T2,T4,T2,cv::gpu::GpuMat(),CV_32FC1);

	cv::gpu::divide(T1,T2,K,1.0,CV_32FC1);
	
	// Ищем смещения
	cv::gpu::multiply(Lx1,K,T1,1.0,CV_32FC1);
	cv::gpu::multiply(Ly1,K,T2,1.0,CV_32FC1);

	// Вычислим радиусы
	cv::gpu::sqr(T1,T3);
	cv::gpu::sqr(T2,T4);
	cv::gpu::add(T3,T4,T3,cv::gpu::GpuMat(),CV_32FC1);
	cv::gpu::sqrt(T3,T4);

	// cv::gpu::GaussianBlur(T4,T4,Size(gauss_kernel_size,gauss_kernel_size),sigma);
	T4.download(Rad);

	Mat Rad_c(img.rows,img.cols,CV_16UC1);
	
	// Сделаем дискретизацию с шагом D
	float D=10;
	Rad.convertTo(Rad_c,CV_16UC1,1.0/D);
	Mat Rad_k(img.rows,img.cols,CV_32FC1);
	Rad_c.convertTo(Rad_k,CV_32FC1,D);
	Rad_k=Rad_k/Rad;
	//---------------------------------

	T1.download(Dx);
	T2.download(Dy);

	Dx=Dx/Rad_k;
	Dy=Dy/Rad_k;
	// Голосование

	//-----------------------------------------------------------------------------
	// Курватура
	//-----------------------------------------------------------------------------
	Mat Curv(img.rows,img.cols,CV_32FC1);
	cv::gpu::sqr(Lx2,T1);
	cv::gpu::sqr(Ly2,T2);
	cv::gpu::multiply(Lxy,Lxy,T3);
	cv::gpu::multiply(T3,2,T3);
	cv::gpu::add(T1,T2,T1);
	cv::gpu::add(T1,T3,T1);
	cv::gpu::sqrt(T1,T1);
	
	// размер ядра фильтра (заметно влияет на детект)
	gauss_kernel_size=9;
	// дисперсия для фильтра Гаусса
	sigma=9;

	cv::gpu::GaussianBlur(T1,T1,Size(gauss_kernel_size,gauss_kernel_size),sigma);

	T1.download(Curv);
	// -------------------------------------------------------------------------

	Accumulator=Mat(img.rows,img.cols,CV_32FC1);
	Accumulator=0;

	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			float X=static_cast<float>(j)+Dx.at<float>(i,j);
			float Y=static_cast<float>(i)+Dy.at<float>(i,j);
			if(X<img.cols && Y<img.rows && X>0 && Y>0)
			{
			// По документу Curv, а не логарифм, но так лучше.
			// В прошлой версии это не правильно считалось
			Accumulator.at<float>(Y,X)+=log(Curv.at<float>(i,j)); 
			}
		}
	}

	//----------------------------------------------
	// Постобработка
	//----------------------------------------------
	// Усредним результаты
	// размер ядра фильтра (заметно влияет на детект)
	gauss_kernel_size=21;
	// дисперсия для фильтра Гаусса
	sigma=21;
	//cv::GaussianBlur(Accumulator,Accumulator,Size(gauss_kernel_size,gauss_kernel_size),sigma);


	// Выделим области центров
	double m,M;
	// Размажем
	cv::GaussianBlur(Accumulator,Accumulator,Size(gauss_kernel_size,gauss_kernel_size),sigma);
	cv::minMaxLoc(Accumulator,&m,&M);
	cv::threshold(Accumulator,Accumulator,M-(M-m)*0.15,1,CV_THRESH_TOZERO);

	// Вроде должно само освобождаться, но надо уточнить
	/*
	Img32F_d.release();
	Lx1.release();
	Ly1.release();
	Lx2.release();
	Ly2.release();
	Lxy.release();
	Lx1sq.release();
	Ly1sq.release();
	K.release();
	T1.release();
	T2.release();
	T3.release();
	T4.release();
	*/
}

//----------------------------------------------------------------------
// MAIN
//----------------------------------------------------------------------
int main(int argc, char * argv[])
{
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
	cascade_gpu.load("haarcascade_frontalface_alt2.xml");
	setlocale(LC_ALL, "Russian");
	Mat frame;

	VideoCapture capture(0);
	capture >> frame;
	Mat img(frame.size(),CV_8UC1);
	Mat img_prev(frame.size(),CV_8UC1);
	Mat Rad(img.rows,img.cols,CV_32FC1);
	Mat Res_full_acc(img.rows,img.cols,CV_32FC1);
	Res_full_acc=0;
	if (capture.isOpened())
	{
		while(true)
		{
			capture >> frame;
			cvtColor(frame,img,CV_BGR2GRAY);
			Mat Res;
			Mat Res_full(img.rows,img.cols,CV_32FC1);
			
			vector<Rect> rects;
			rects=detect_faces(img);
			Res_full=0;
			if(rects.size()>0)
			{
				rects[0].height/=2;
				rects[0].y+=rects[0].height/4;

			cv::rectangle(frame,rects[0],CV_RGB(255,0,0));
			IsofoteCurvatureDetector(img(Rect(rects[0])),Res,Rad);
			Res.copyTo(Res_full(Rect(rects[0])));
			}
			cv::normalize(Res_full,Res_full,0,1,CV_MINMAX);
			
			cv::accumulateWeighted(Res_full,Res_full_acc,0.995);
			imshow("Res",Res_full_acc);
			DetectContour(Res_full_acc, img,Rad);
			
			int c = waitKey(10);
			if( (char)c == 27 ) 
			{ 
				break; 
			} 
		}
	}

	return 0;
}