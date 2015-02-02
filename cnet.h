#pragma once
#include <vector>
#include <string.h>
#include "opencv/cv.h"
using namespace std;

//��������w��ʾweights��b��ʾbias��s��ʾSLayer��c��ʾCLayer��f��ʾFLayer
typedef struct SLayer
{
	int transfFunc;//�����������ͣ�0��purelin��1��tran_sig
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
	int transfFunc;//�����������ͣ�0��purelin��1��tran_sig
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
	int transfFunc;//�����������ͣ�0��purelin��1��tran_sig
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
	int m_InputWidth;//����ͼ��Ŀ��
	int m_InputHeight;
public:
	CNet(void);
	~CNet(void);
	bool loadParam(string file);//����ѵ������
	void fProp(CvMat* img, float* pos);//ǰ�򴫲�
	void releaseLayers();//�ͷŸ���ļ�����
private:
	int m_numLayers;//����������
	int m_numSLayers;//�²�������
	int m_numCLayers;//��������
	int m_numFLayers;//ȫ���Ӳ����
	int m_numInputs;//�����������������һ����1����ʾһ��ͼ��
	int m_numOutputs;//�������
	PSLayer m_SLayer;//�²�������
	PCLayer m_CLayer;//��������
	PFLayer m_FLayer;//ȫ���Ӳ����
	void subsample(CvMat* src, CvMat* dst);//�²�������
	void sigmoidFunc(CvMat* src, CvMat* dst);//˫�����к���
	void filter(CvMat* src, CvMat* dst, CvMat* ker);//������
	
};

