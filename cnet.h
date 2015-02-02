#pragma once
#include <vector>
#include <string.h>
#include "opencv/cv.h"
using namespace std;

//命名规则：w表示weights，b表示bias，s表示SLayer，c表示CLayer，f表示FLayer
typedef struct SLayer
{
	int transfFunc;//激励函数类型，0：purelin，1：tran_sig
	int numFMaps;
	vector<float> WS;
	vector<float> BS;
	//vector<CvMat*> YS;
	vector<CvMat*> XS;
	//vector<CvMat*> SS;
	~SLayer(void){
		WS.clear();
		BS.clear();

		if(XS.size()!=0)
		{
			for (vector<CvMat*>::iterator it = XS.begin(); it != XS.end(); it++) 
			if (NULL != *it) 
			{
				cvReleaseMat(&(*it)); 
			}
			XS.clear();
		}

	}
}SLayer,*PSLayer;

typedef struct CLayer
{
	int transfFunc;//激励函数类型，0：purelin，1：tran_sig
	int numFMaps;
	int numKernPerFMaps;
	int cKernWidth;
	int cKernHeight;
	vector<CvMat*> WC;
	vector<float> BC;
	CvMat* conMap;
	vector<CvMat*> YC;
	vector<CvMat*> XC;
	~CLayer(void){
		BC.clear();
		WC.clear();
		cvReleaseMat(&conMap);
		if(YC.size()!=0)
		{
			for (vector<CvMat*>::iterator it = YC.begin(); it != YC.end(); it++) 
			if (NULL != *it) 
			{
				cvReleaseMat(&(*it)); 
			}
			YC.clear();
		}
		if(XC.size()!=0)
		{
			for (vector<CvMat*>::iterator it = XC.begin(); it != XC.end(); it++) 
			if (NULL != *it) 
			{
				cvReleaseMat(&(*it)); 
			}
			XC.clear();
		}

		for (vector<CvMat*>::iterator it = WC.begin(); it != WC.end(); it++) 
		if (NULL != *it) 
		{
			cvReleaseMat(&(*it)); 
		}
	}
}CLayer,*PCLayer;

typedef struct FLayer
{
	int transfFunc;//激励函数类型，0：purelin，1：tran_sig
	CvMat* WF;
	CvMat* BF;
	CvMat* XF;
	~FLayer(void){
		cvReleaseMat(&WF);
		cvReleaseMat(&BF);
		if(XF)
			cvReleaseMat(&XF);
	}
}FLayer,*PFLayer;

class CNet
{
public:
	int m_InputWidth;//输入图像的宽高
	int m_InputHeight;
public:
	CNet(void);
	~CNet(void);
	bool loadParam(string file);//加载训练参数
	void fProp(CvMat* img, float* pos);//前向传播
	void releaseLayers();//释放各层的计算结果
private:
	int m_numLayers;//卷积网络层数
	int m_numSLayers;//下采样层数
	int m_numCLayers;//卷积层层数
	int m_numFLayers;//全连接层层数
	int m_numInputs;//卷积网络输入数量，一般是1，表示一幅图像
	int m_numOutputs;//输出个数
	PSLayer m_SLayer;//下采样层数
	PCLayer m_CLayer;//卷积层层数
	PFLayer m_FLayer;//全连接层层数
	void subsample(CvMat* src, CvMat* dst);//下采样函数
	void sigmoidFunc(CvMat* src, CvMat* dst);//双曲正切函数
	void filter(CvMat* src, CvMat* dst, CvMat* ker);//计算卷积
	
};

