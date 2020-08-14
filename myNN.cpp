#include <stdio.h>  //输出文本信息					printf
#include <stdlib.h> //生成随机数、动态开辟内存 		rand、malloc
#include <time.h>   //随机数种子						srand(time(NULL))
#include <string.h> //memset快速按字节初始化数组		memset
#include <math.h>   //指数函数						exp(x) 

#define TRAIN_IMAGES_NUM 60000
#define TEST_IMAGES_NUM 10000
#define LEARNING_RATE 0.3
#define LAYERS_NUM 4
#define INPUT_LAYERS_SIZE 784
#define HIDDEN_LAYERS_SIZE 50
#define OUTPUT_LAYERS_SIZE 10
#define EPOCHS 50

//定义激活函数
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

//生成-1~1的随机浮点数
double randomWeightOrBias()
{
	return  2 * (double)rand() / RAND_MAX - 1.0;
}

//神经元结构体定义
typedef struct Neuron
{
	double b;//偏置值
	double z;//加权和
	double a;//激活值
	double* w;//指向与前一层所有神经元相连的权重数组的指针
	double pdC2B;//损失函数对当前神经元偏置的偏导数 ∂C/∂b
} NEURON;

//神经层结构体定义
typedef struct Layer
{
	int numOfNeurons;//该层的神经元数
	NEURON* neurons;//指向当前层神经元数组的指针
}LAYER;

//神经网络结构体定义
typedef struct NNet
{
	int numOfLayers;//神经网络中的层数
	LAYER* layers;//指向神经网络层 数组的指针
}NNET;

//arrOfNumOfNeuronsOfEachLayers为存储各层神经元个数的数组
//包括给各神经层分配内存，给各层神经元分配内存，给各权重网络分配内存并且随机化
void initNeuronNet(NNET* nnet, int numOfLayers, int* arrOfNumOfNeuronsOfEachLayers)
{
	nnet->numOfLayers = numOfLayers;
	//给layers数组动态分配内存
	nnet->layers = (LAYER*)malloc(sizeof(LAYER) * numOfLayers);
	//给每层的神经元数组动态分配内存
	for (int i = 0; i < numOfLayers; i++)
	{
		nnet->layers[i].numOfNeurons = arrOfNumOfNeuronsOfEachLayers[i];
		nnet->layers[i].neurons = (NEURON*)malloc(sizeof(NEURON) * arrOfNumOfNeuronsOfEachLayers[i]);
	}

	//从第二层开始初始化权重
	for (int i = 1; i < nnet->numOfLayers; i++)
	{
		

		for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
		{
			//每层的每个神经元的权值数组 动态分配内存 大小为上一层的神经元个数
			nnet->layers[i].neurons[j].w = (double*)malloc(sizeof(double) * nnet->layers[i - 1].numOfNeurons);
			//每层的每个神经元的偏置随机化
			nnet->layers[i].neurons[j].b = randomWeightOrBias();
			for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
			{
				double weight = randomWeightOrBias();
				nnet->layers[i].neurons[j].w[k] = weight;
			}
		}
	}
}

//输入层输入数据正向传播
void forwardPropWithInput(NNET* nnet, double* inputs)
{
	for (int i = 0; i < nnet->layers[0].numOfNeurons; i++)
	{
		//输入层各神经元激活值初始化
		nnet->layers[0].neurons[i].a = inputs[i];
	}

	//从第二层开始初始化
	for (int i = 1; i < nnet->numOfLayers; i++)
	{

		for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
		{
			//加权和
			double z = 0;
			for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
			{
				double weight = nnet->layers[i].neurons[j].w[k];
				z += nnet->layers[i - 1].neurons[k].a * weight;
			}
			nnet->layers[i].neurons[j].z = z + nnet->layers[i].neurons[j].b;
			nnet->layers[i].neurons[j].a = sigmoid(nnet->layers[i].neurons[j].z);
		}

	}
}


//根据期望值 反向传播 并且更新权值网络
void backProp(NNET* nnet, double* targets)
{
	//num为最后一层的神经元数
	int numOfNeuronsOfLastLayer = nnet->layers[nnet->numOfLayers - 1].numOfNeurons;
	//i为最后一层神经元数组下标
	for (int i = 0; i < numOfNeuronsOfLastLayer; i++)
	{
		//activation等于最后一层神经元的激活值
		double activation = nnet->layers[nnet->numOfLayers - 1].neurons[i].a;
		//最后一层每个神经元的误差值(Cost对权重b的偏导数) pdC2B = sigmoid(z) * ( 1 - sigmoid(z) ) * 2 * (y - a) / OUTPUT_LAYER_SIZE
		nnet->layers[nnet->numOfLayers - 1].neurons[i].pdC2B = activation * (1 - activation) * (targets[i] - activation) * 2 / OUTPUT_LAYERS_SIZE;
		//更新权值
		nnet->layers[nnet->numOfLayers - 1].neurons[i].b += LEARNING_RATE * nnet->layers[nnet->numOfLayers - 1].neurons[i].pdC2B;
	}

	//i为当前层，从后往前
	for (int i = nnet->numOfLayers - 1; i > 0; i--)
	{
		//j从倒数第二层的神经元数开始，j为前一层的神经元数
		for (int j = 0; j < nnet->layers[i - 1].numOfNeurons; j++)
		{
			double sumOfPdOfActivationOfPreviousLayer = 0;
			//k为当前层神经元数
			for (int k = 0; k < nnet->layers[i].numOfNeurons; k++)
			{
				//对前一层激活值的偏导数
				sumOfPdOfActivationOfPreviousLayer += nnet->layers[i].neurons[k].w[j] * nnet->layers[i].neurons[k].pdC2B;

				//对当前层的权重偏导数更新  更新的权重=学习率*累加(对当前层权重的偏导数(=对偏置b的偏导数*前一层的激活值))
				nnet->layers[i].neurons[k].w[j] += LEARNING_RATE * nnet->layers[i].neurons[k].pdC2B * nnet->layers[i - 1].neurons[j].a;

			}
			//前一层神经元的激活值
			double activation = nnet->layers[i - 1].neurons[j].a;
			//前一层神经元对偏置的导数的更新
			nnet->layers[i - 1].neurons[j].pdC2B = activation * (1 - activation) * sumOfPdOfActivationOfPreviousLayer;
			nnet->layers[i - 1].neurons[j].b += LEARNING_RATE * nnet->layers[i - 1].neurons[j].pdC2B;
		}

	}
}


//存储模型权重数据
void saveModelData(NNET* nnet, FILE* fpModel)
{
	fpModel = fopen("./model.dat", "w+b");
	//从第二层开始才有权重网络
	for (int i = 1; i < nnet->numOfLayers; i++)
	{
		for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
		{
			for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
			{
				fwrite(&(nnet->layers[i].neurons[j].w[k]), sizeof(double), 1, fpModel);
			}
			fwrite(&(nnet->layers[i].neurons[j].b), sizeof(double), 1, fpModel);
		}
	}
	fclose(fpModel);
}


//读取模型权重数据
void readModelData(NNET* nnet, FILE* fpModel)
{
	fpModel = fopen("./model.dat", "rb");
	//从第二层开始才有权重网络
	for (int i = 1; i < nnet->numOfLayers; i++)
	{
		for (int j = 0; j < nnet->layers[i].numOfNeurons; j++)
		{
			for (int k = 0; k < nnet->layers[i - 1].numOfNeurons; k++)
			{
				fread(&(nnet->layers[i].neurons[j].w[k]), sizeof(double), 1, fpModel);
			}
			fread(&(nnet->layers[i].neurons[j].b), sizeof(double), 1, fpModel);
		}
	}
	fclose(fpModel);
}


//从数据集中读图片到数组中
void initBufferArrOfImg(FILE* fpImg, double** bufferArr, int numOfImgs)
{
	unsigned char* tmpBufferOfImg = (unsigned char*)malloc(sizeof(unsigned char) * INPUT_LAYERS_SIZE);

	for (int i = 0; i < numOfImgs; i++)
	{
		fread(tmpBufferOfImg, sizeof(unsigned char), INPUT_LAYERS_SIZE, fpImg);

		for (int j = 0; j < INPUT_LAYERS_SIZE; j++)
		{
			bufferArr[i][j] = tmpBufferOfImg[j] / 255.0;
		}
	}
	free(tmpBufferOfImg);

}


//从数据集中读标签到数组中
void initBufferArrOfLabel(FILE* fpLabel, int* bufferArr, int numOfLabels)
{
	unsigned char* tmpBufferOfLabel = (unsigned char*)malloc(sizeof(unsigned char) * numOfLabels);
	//memset(tmpBufferOfLabel, 0, sizeof(unsigned char) * TRAIN_IMAGES_NUM);

	fread(tmpBufferOfLabel, sizeof(unsigned char), numOfLabels, fpLabel);
	for (int i = 0; i < numOfLabels; i++)
	{
		bufferArr[i] = tmpBufferOfLabel[i];
	}
	free(tmpBufferOfLabel);

}


//验证模型对测试集的正确率
double accuracyRate(NNET* nnet, double** tBufferArrOfImg, int* tBufferArrOfLabel)
{
	int cntRight = 0;
	for (int i = 0; i < TEST_IMAGES_NUM; i++)
	{
		double tInputs[INPUT_LAYERS_SIZE];
		for (int j = 0; j < INPUT_LAYERS_SIZE; j++)
		{
			tInputs[j] = tBufferArrOfImg[i][j];
		}

		forwardPropWithInput(nnet, tInputs);

		double max = nnet->layers[nnet->numOfLayers - 1].neurons[0].a;
		int guessNum = 0;
		for (int j = 0; j < OUTPUT_LAYERS_SIZE; j++)
		{
			if (nnet->layers[nnet->numOfLayers - 1].neurons[j].a > max)
			{
				max = nnet->layers[nnet->numOfLayers - 1].neurons[j].a;
				guessNum = j;
			}
		}
	
		if (guessNum == tBufferArrOfLabel[i])
		{
			cntRight++;
		}
	}
	return (double)cntRight / TEST_IMAGES_NUM;
}

int main()
{
	/*-----------------------读训练集文件------------------------*/
	int magicNum, picNum, pixelRow, pixelCol;
	int lMagicNum, labelNum;

	FILE* fpImg = fopen("./train-images.idx3-ubyte", "rb");
	if (!fpImg)
	{
		printf("unable open train-images.idx3-ubyte");
	}
	else
	{
		fread(&magicNum, sizeof(int), 1, fpImg);
		fread(&picNum, sizeof(int), 1, fpImg);
		fread(&pixelRow, sizeof(int), 1, fpImg);
		fread(&pixelCol, sizeof(int), 1, fpImg);
	}

	FILE* fpLable = fopen("./train-labels.idx1-ubyte", "rb");
	if (!fpLable)
	{
		printf("unable open train-labels.idx1-ubyte");
	}
	else
	{
		fread(&lMagicNum, sizeof(int), 1, fpLable);
		fread(&labelNum, sizeof(int), 1, fpLable);
	}
	/*----------------------------------------------------------*/


	/*----------------------读测试集文件-------------------------*/

	/*
		t10k-images.idx3-ubyte
		t10k-labels.idx1-ubyte
	*/
	int tMagicNum, tPicNum, tPixelRow, tPixelCol;
	int tLMagicNum, tLabelNum;

	FILE* tFpImg = fopen("./t10k-images.idx3-ubyte", "rb");
	if (!tFpImg)
	{
		printf("unable open t10k-images.idx3-ubyte");
	}
	else
	{
		fread(&tMagicNum, sizeof(int), 1, tFpImg);
		fread(&tPicNum, sizeof(int), 1, tFpImg);
		fread(&tPixelRow, sizeof(int), 1, tFpImg);
		fread(&tPixelCol, sizeof(int), 1, tFpImg);
	}

	FILE* tFpLable = fopen("./t10k-labels.idx1-ubyte", "rb");
	if (!tFpLable)
	{
		printf("unable open t10k-labels.idx1-ubyte");
	}
	else
	{
		fread(&tLMagicNum, sizeof(int), 1, tFpLable);
		fread(&tLabelNum, sizeof(int), 1, tFpLable);
	}
	
	/*------------------------------------------------------------*/


	//用来存储训练集中图片灰度值的数组
	double** bufferArrOfImg = (double**)malloc(sizeof(double*) * TRAIN_IMAGES_NUM);
	for (int i = 0; i < TRAIN_IMAGES_NUM; i++)
	{
		bufferArrOfImg[i] = (double*)malloc(sizeof(double) * INPUT_LAYERS_SIZE);
	}
	initBufferArrOfImg(fpImg, bufferArrOfImg, TRAIN_IMAGES_NUM);

	//用来存储训练集中标签的数组
	int* bufferArrOfLabel = (int*)malloc(sizeof(int) * TRAIN_IMAGES_NUM);
	initBufferArrOfLabel(fpLable, bufferArrOfLabel, TRAIN_IMAGES_NUM);



	//用来存储测试集中图片灰度值的数组
	double** tBufferArrOfImg = (double**)malloc(sizeof(double*) * TEST_IMAGES_NUM);
	for (int i = 0; i < TEST_IMAGES_NUM; i++)
	{
		tBufferArrOfImg[i] = (double*)malloc(sizeof(double) * INPUT_LAYERS_SIZE);
	}
	initBufferArrOfImg(tFpImg, tBufferArrOfImg, TEST_IMAGES_NUM);

	//用来存储测试集中标签的数组
	int* tBufferArrOfLabel = (int*)malloc(sizeof(int) * TEST_IMAGES_NUM);
	initBufferArrOfLabel(tFpLable, tBufferArrOfLabel, TEST_IMAGES_NUM);





	//给网络分配内存空间
	NNET* net = (NNET*)malloc(sizeof(NNET));


	//存放各层 神经元个数 的 数组
	int arrOfNumOfNeuronsOfEachLayers[LAYERS_NUM] = { INPUT_LAYERS_SIZE,
													HIDDEN_LAYERS_SIZE,
													HIDDEN_LAYERS_SIZE,
													OUTPUT_LAYERS_SIZE };


	//初始化随机数种子
	srand((unsigned int)time(NULL));


	//初始化神经网络内存空间
	initNeuronNet(net, LAYERS_NUM, arrOfNumOfNeuronsOfEachLayers);


	//初始化用来保存模型数据的文件指针
	FILE* fpModel = NULL;



	for (int k = 0; k < EPOCHS; k++)
	{
		for (int j = 0; j < TRAIN_IMAGES_NUM; j++)
		{
			double inputs[INPUT_LAYERS_SIZE];
			for (int i = 0; i < INPUT_LAYERS_SIZE; i++)
			{
				inputs[i] = bufferArrOfImg[j][i];
			}

			double targets[OUTPUT_LAYERS_SIZE];
			memset(targets, 0, sizeof(double)* OUTPUT_LAYERS_SIZE);
			targets[bufferArrOfLabel[j]] = 1.0;


			//输入层数据前向传播
			forwardPropWithInput(net, inputs);
			//根据目标标签向量反向传播
			backProp(net, targets);


			if ((j + 1) % 10000 == 0)
			{
				printf("epoch:%d  ,  index of Image is %d  ,  label of image is %d\n", k + 1, j + 1, bufferArrOfLabel[j]);
				for (int i = 0; i < net->layers[3].numOfNeurons; i++)
				{
					printf("%d:   %.20lf\n", i, net->layers[3].neurons[i].a);
				}
				printf("\n");


				/*printf("-------------------testImg:7------------------------\n\n");
				forwardPropWithInput(net, tbuffer);
				for (int i = 0; i < net->layers[3].numOfNeurons; i++)
				{
					printf("%d:   %.20lf\n", i, net->layers[3].neurons[i].a);
				}
				printf("\n");*/

			}
		}
		saveModelData(net, fpModel);
		printf("accuracy rate is %lf%%\n", 100 * accuracyRate(net, tBufferArrOfImg, tBufferArrOfLabel));
	}



	free(bufferArrOfImg);
	free(bufferArrOfLabel);
	free(tBufferArrOfImg);
	free(tBufferArrOfLabel);

	fclose(fpImg);
	fclose(fpLable);
	fclose(tFpImg);
	fclose(tFpLable);

	getchar();
	return 0;
}