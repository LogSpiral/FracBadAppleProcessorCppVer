#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <execution>
#include <sstream>
#include <string>
#include "time.h"
#include <filesystem>
#include <cmath>
using namespace cv;
using namespace std::filesystem;
//分形烂苹果生成器，CPP-CPU版本绝赞制作中！！
//GPU版本因为螺线还不会CUDA所以咕咕咕了
void logCurrentTime()
{
	// 获取当前时间点
	auto now = std::chrono::system_clock::now();
	// 转换为time_t格式，以便使用ctime库
	std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
	// 将time_t转换为本地时间
	std::tm now_tm;
	localtime_s(&now_tm, &now_time_t);
	std::cout << "当前时间是: " << std::put_time(&now_tm, "%Y-%m-%d %X") << "\n";
}
int scale = 4096;
Mat dataSet;
Mat fracImage;
Vec3b black = Vec3b(0, 0, 0);
Vec3b white = Vec3b(255, 255, 255);
uchar standard_BlackWhite;
/*void processorTemplate(int i, int j, Vec3b& color, Mat img)
{

}*/
void processorBlackWhite(Vec3b& color)
{
	color = (color[0] / 3 + color[1] / 3 + color[2] / 3) > standard_BlackWhite ? white : black;
	//auto orig = color;
	//double value = color[0] / 255.0 + color[1] / 255.0 + color[2] / 255.0;
	//uchar c = value > 0.33 ? 255 : 0;
	//color = Vec3b(c, c, c);
}
template<typename T>
T clamp(T value, T min, T max)
{
	if (value > max)return max;
	if (value < min)return min;
	return value;
}
int lengthSquared(Vec2i vec)
{
	return vec[0] * vec[0] + vec[1] * vec[1];
}
Vec3b IntToColor(int p)
{
	return Vec3b((uchar)p, (uchar)(p >> 8), (uchar)(p >> 16));

}
void processorESSEDT(Mat img)
{
	//初始化

	int width = img.cols;
	int height = img.rows;
	Vec2i** deltaS = new Vec2i * [height];
	Vec2i* data = new Vec2i[height * width];
	for (int i = 0; i < height; ++i) {
		deltaS[i] = &data[i * width];
	}
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{

			deltaS[i][j] = (img.at<Vec3b>(i, j)[0] > 20) ? Vec2i(0, 0) : Vec2i(width, height);
		}


	// 第一个像素(左上)
	{
		bool flag = true;
		int counter = 0;//计数器，表示对当前像素查找的次数
		Vec2i unit = Vec2i(0, 0);
		float dist = 0;
		while (flag)
		{
			int x = counter % scale;
			int y = counter / scale * 2;
			Vec3b xData = dataSet.at<Vec3b>(y, x);//从数据图中获得xy偏移量
			Vec3b yData = dataSet.at<Vec3b>(y + 1, x);
			unit = Vec2i((xData[2] * 256 + xData[1]) * 256 + xData[0], (yData[2] * 256 + yData[1]) * 256 + yData[0]);//生成偏移向量
			for (int n = 0; n < 2; n++)
			{
				Vec2i _unit = unit;
				//以下三行对应三个对称操作，由一个偏移向量生成等模长的三个
				if (n > 0) _unit = Vec2i(_unit[1], _unit[0]);

				//查询格点，如果是白色像素就停止(只有两个状态，所以我直接x>0了
				if (img.at<Vec3b>(clamp(_unit[1], 0, height - 1), clamp(_unit[0], 0, width - 1))[0] > 0)
				{
					flag = false;//停止当前像素的查找
					dist = lengthSquared(unit);//记录该像素到最近白色像素的距离的平方
					unit = _unit;
					break;
				}
			}
			counter++;//查询次数自增
		}
		deltaS[0][0] = unit;
	}
	// 上到下扫描
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			if (deltaS[i][j] == Vec2i(0, 0)) continue;
			Vec2i& cur = deltaS[i][j];
			if (j != 0)
			{
				Vec2i tar = deltaS[i][j - 1] + Vec2i(-1, 0);
				if (lengthSquared(tar) < lengthSquared(cur))
				{
					cur = tar;
				}

			}
			if (i != 0)
			{
				Vec2i tar = deltaS[i - 1][j] + Vec2i(0, -1);
				if (lengthSquared(tar) < lengthSquared(cur))
				{
					cur = tar;
				}
				if (j != 0)
				{
					tar = deltaS[i - 1][j - 1] + Vec2i(-1, -1);
					if (lengthSquared(tar) < lengthSquared(cur))
					{
						cur = tar;
					}
				}
				if (j != width - 1)
				{
					tar = deltaS[i - 1][j + 1] + Vec2i(1, -1);
					if (lengthSquared(tar) < lengthSquared(cur))
					{
						cur = tar;
					}
				}
			}
		}
	// 第一个像素(右下)
	{
		bool flag = true;
		int counter = 0;//计数器，表示对当前像素查找的次数
		Vec2i unit = Vec2i(0, 0);
		float dist = 0;
		while (flag)
		{
			int x = counter % scale;
			int y = counter / scale * 2;
			Vec3b xData = dataSet.at<Vec3b>(y, x);//从数据图中获得xy偏移量
			Vec3b yData = dataSet.at<Vec3b>(y + 1, x);
			unit = Vec2i((xData[2] * 256 + xData[1]) * 256 + xData[0], (yData[2] * 256 + yData[1]) * 256 + yData[0]);//生成偏移向量

			for (int n = 0; n < 2; n++)
			{
				Vec2i _unit = unit;
				//以下三行对应三个对称操作，由一个偏移向量生成等模长的三个
				if (n > 0) _unit = Vec2i(_unit[1], _unit[0]);
				_unit *= -1;
				_unit += Vec2i(width - 1, height - 1);

				//查询格点，如果是白色像素就停止(只有两个状态，所以我直接x>0了
				if (img.at<Vec3b>(clamp(_unit[1], 0, height - 1), clamp(_unit[0], 0, width - 1))[0] > 0)
				{
					flag = false;//停止当前像素的查找
					dist = lengthSquared(unit);//记录该像素到最近白色像素的距离的平方
					unit = _unit;
					break;
				}
			}
			counter++;//查询次数自增
		}
		Vec2i tar = unit;// -new Vector2(width - 1, height - 1);
		if (lengthSquared(tar) < lengthSquared(deltaS[height - 1][width - 1]))
			deltaS[height - 1][width - 1] = tar;
	}
	// 下到上扫描
	for (int i = height - 1; i >= 0; i--)
		for (int j = width - 1; j >= 0; j--)
		{
			if (deltaS[i][j] == Vec2i(0, 0)) continue;
			Vec2i& cur = deltaS[i][j];
			if (j != width - 1)
			{
				Vec2i tar = deltaS[i][j + 1] + Vec2i(1, 0);
				if (lengthSquared(tar) < lengthSquared(cur))
				{
					cur = tar;
				}

			}
			if (i != height - 1)
			{
				Vec2i tar = deltaS[i + 1][j] + Vec2i(0, 1);
				if (lengthSquared(tar) < lengthSquared(cur))
				{
					cur = tar;
				}
				if (j != 0)
				{
					tar = deltaS[i + 1][j - 1] + Vec2i(-1, 1);
					if (lengthSquared(tar) < lengthSquared(cur))
					{
						cur = tar;
					}
				}
				if (j != width - 1)
				{
					tar = deltaS[i + 1][j + 1] + Vec2i(1, 1);
					if (lengthSquared(tar) < lengthSquared(cur))
					{
						cur = tar;
					}
				}
			}
		}
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
		{
			img.at<Vec3b>(i, j) = IntToColor(lengthSquared(deltaS[i][j]));//用像素来记录距离信息
		}

	delete[] data;  // 释放整个内存块
	delete[] deltaS;  // 释放行指针数组
}
void processorFractal(double angle, Vec3b& color)
{
	int distSqr = color[0] + 255 * (color[1] + 255 * color[2]);//把像素信息转距离信息
	Vec2d orig = Vec2d(-32 / 9.0, -2.0);//缩放中心
	//double angle = atan2(i - height * .5, j - width * .5) * 4;
	Vec2d vec = Vec2d(cos(angle), sin(angle)) * 0.5;
	angle *= 2;
	vec -= Vec2d(cos(angle), sin(angle)) * 0.25;
	vec *= 0.95 + clamp(sqrt(distSqr) / 8000.0, 0.0, 1000.0) * 64.0;
	vec -= orig;
	vec *= 270;
	color = fracImage.at<Vec3b>(clamp((int)vec[1], 0, 1079), clamp((int)vec[0], 0, 1919));
}
class PixelOperation_BlackWhite : public cv::ParallelLoopBody {
public:
	PixelOperation_BlackWhite(Mat& _img) : img(_img) {}

	void operator()(const cv::Range& range) const {
		for (int i = range.start; i < range.end; i++) {
			for (int j = 0; j < img.cols; j++) {
				processorBlackWhite(img.at<Vec3b>(i, j));
			}
			//std::cout << "Processed row " << i << std::endl;
		}
	}

private:
	Mat& img;
};
class PixelOperation_Fractal : public cv::ParallelLoopBody {
public:
	PixelOperation_Fractal(Mat& _img) : img(_img) {}

	void operator()(const cv::Range& range) const {
		int width = img.cols;
		int height = img.rows;
		for (int i = range.start; i < range.end; i++) {
			for (int j = 0; j < width; j++) {
				processorFractal(atan2(i - height * .5, j - width * .5) * 4, img.at<Vec3b>(i, j));
			}
			//std::cout << "Processed row " << i << std::endl;
		}
	}

private:
	Mat& img;
};
int main(int argc, char* argv[])
{
	if (argc == 1)
	{
		std::cout << "请尝试拖入一些文件来给exe处理吧\n";
		std::cin.get();
		return 0;
	}
	path p = path(argv[0]).parent_path();
	dataSet = imread((p / "dataSet4096.png").string(), IMREAD_COLOR);
	fracImage = imread((p / "WallPaper_FractalMandelbort.png").string(), IMREAD_COLOR);

	std::cout << "请输入相应数字来执行相应功能\n";
	std::cout << "0 将画面进行裁切(3840x2160→2880x2160)\n";
	std::cout << "1 将画面阈值处理，亮度高于0.33的改成白色，否则是黑色，亮度∈[0,1]\n";
	std::cout << "2 将 黑白 画面进行距离场处理，计算每个像素到最近的白色像素的距离的平方并且以颜色形式存储\n";
	std::cout << "3 将 距离场 图进行分形映射处理\n";
	std::cout << "4 除了裁切以外的一条龙服务，适用于大多数图(图片亮度太低会在1处理成纯黑然后2炸掉)\n";
	int index;
	std::cin >> index;
	if (index < 0 || index > 4)
	{
		std::cout << "不认识的数字呢\n";
		std::cin.get();
		std::cin.get();
		return 0;
	}
	int standard_BW = 0;
	if (index == 1 || index == 4)
	{
		std::cout << "请输入黑白阈值(0-255)\n";
		std::cin >> standard_BW;
		standard_BlackWhite = (uchar)standard_BW;
	}
	std::cout << "开始，";
	logCurrentTime();
	std::cout << "请耐心等待处理\n";
	//std::cout << "请耐心等待处理，多于1000张时每处理100张会输出一次进度，否则多于10张时每10张输出一次进度\n";
	int counter = 0;
	auto processor = [index, argc, &counter](char* charpath)
		{
			/*std::ostringstream ss;
			ss << path << "," << counter++ << "\n";
			std::string merged = ss.str();
			std::cout << merged;*/
			counter++;
			int c = counter;
			Mat img = imread(charpath, IMREAD_COLOR);
			switch (index)
			{
			case 0:
				std::cout << "我懒得做这个裁切了，这个是因为烂苹果是4：3而那个4k 60帧的是16：9我需要裁切一下才存在的\n";
				//std::cout << charpath << img.cols << "x" << img.rows << "\n";
				std::cin.get();
				std::cin.get();
				return 0;
			case 1:
				parallel_for_(Range(0, img.rows), PixelOperation_BlackWhite(img));
				//for (int i = 0; i < img.rows; i++)
				//	for (int j = 0; j < img.cols; j++)
				//	{
				//		processorBlackWhite(img.at<Vec3b>(i, j));
				//	}
				break;
			case 2:
				processorESSEDT(img);
				break;
			case 3:
				parallel_for_(Range(0, img.rows), PixelOperation_Fractal(img));
				//for (int i = 0; i < img.rows; i++)
				//	for (int j = 0; j < img.cols; j++)
				//	{
				//		processorFractal(j, i, img.cols, img.rows, img.at<Vec3b>(i, j));
				//	}
				break;
			case 4:
				parallel_for_(Range(0, img.rows), PixelOperation_BlackWhite(img));
				processorESSEDT(img);
				parallel_for_(Range(0, img.rows), PixelOperation_Fractal(img));
				break;
			default:
				break;
			}
			path curpath = path(charpath);
			path sourceDir = curpath.parent_path();
			path resultDir = sourceDir / "Result_Cpp";
			if (!exists(resultDir)) {
				create_directories(resultDir);
			}
			//if (argc > 10 && c % (argc > 1000 ? 100 : 10) == 0)
			//{
			//	std::ostringstream ss;
			//	ss << (c * 100.0 / argc) << "%\n";
			//	std::string merged = ss.str();
			//	std::cout << merged;
			//}
			imwrite((resultDir / curpath.filename()).string(), img);

			//std::ostringstream _ss;
			//_ss << charpath << "_result.png";
			//std::string _merged = _ss.str();
			//imwrite(_merged, img);
		};
	std::vector<char*> data;

	char** begin = argv + 1;
	char** end = argv + argc;
	std::copy(begin, end, std::back_inserter(data));

	std::vector<std::thread> threads;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < data.size(); ++i) {
		threads.emplace_back(processor, data[i]);
	}
	// 等待所有线程完成
	for (auto& thread : threads) {
		thread.join();
	}
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = stop - start;
	std::cout << "Function took " << duration.count() << " milliseconds." << std::endl;

	std::cout << "处理成功，已处理" << counter << "个文件，请查看它们自己目录下的Result_Cpp文件夹\n";
	std::cout << "结束，";
	logCurrentTime();
	std::cout << "输入任意按键退出\n";
	std::cin.get();
	std::cin.get();


}