#include "stdafx.h"
#include "common.h"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <random>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>


#define BLOCK_SIZE 8

using namespace std;

double dataLuminance[8][8] = {
	{16, 11, 10, 16, 24, 40, 51, 61},
	{12, 12, 14, 19, 26, 58, 60, 55},
	{14, 13, 16, 24, 40, 57, 69, 56},
	{14, 17, 22, 29, 51, 87, 80, 62},
	{18, 22, 37, 56, 68, 109, 103, 77},
	{24, 35, 55, 64, 81, 104, 113, 92},
	{49, 64, 78, 87, 103, 121, 120, 101},
	{72, 92, 95, 98, 112, 100, 103, 99}
};


bool isInside(Mat img, int i, int j) {
	return (i >= 0 && i < img.rows) && (j >= 0 && j < img.cols);
}
Mat_<float> discreteCosine(Mat_<float> block)
{
	Mat_<float> block2(block.rows,block.cols);
	float ci, cj, aux, sum;
	for (int i = 0; i < block.rows; i++)
		for (int j = 0; j < block.cols; j++)
		{
			if (i == 0)
				ci = 1/sqrt(8);
			else ci = sqrt(2)/sqrt(8);

			if (j == 0)
				cj = 1/sqrt(8);
			else cj = sqrt(2)/sqrt(8);

			sum = 0;
			for (int u=0;u<block.rows;u++)
				for (int v = 0; v < block.cols; v++)
				{
					aux = block(v, u) * cos((2 * v + 1) * i * PI / (2 * block.rows)) *
						cos((2 * u + 1) * j * PI / (2 * block.cols));
					sum += aux;
				}
			float summing = ci * cj * sum;
			block2(i, j) = round(summing);
		}

	return block2;
}
std::vector<int> zigZagTraversal(const cv::Mat& matrix) {
    std::vector<int> traversal;
    int numRows = matrix.rows;
    int numCols = matrix.cols;
    int rowIndex = 0;
    int colIndex = 0;
    int direction = 1;

    while (rowIndex < numRows && colIndex < numCols) {
        traversal.push_back(matrix.at<int>(rowIndex, colIndex));

        // Determine the next cell based on the current direction
        if (direction == 1) {
            if (colIndex == numCols - 1) {
                ++rowIndex;
                direction = -1;
            } else if (rowIndex == 0) {
                ++colIndex;
                direction = -1;
            } else {
                --rowIndex;
                ++colIndex;
            }
        } else {
            if (rowIndex == numRows - 1) {
                ++colIndex;
                direction = 1;
            } else if (colIndex == 0) {
                ++rowIndex;
                direction = 1;
            } else {
                ++rowIndex;
                --colIndex;
            }
        }
    }

    return traversal;
}
std::vector<int> eliminateZeros(vector<int> vec)
{
	int i=vec.size()-1;
	while (vec.at(i) == 0&&i>0)
	{
		vec.pop_back();
		i = vec.size() - 1;
	}
	return vec;
}
Mat_<int> createMatrix(vector<int> vec)
{
	Mat_<int> result(BLOCK_SIZE, BLOCK_SIZE);
	int row = 0, col = 0; // Starting position
	for (int i = 0; i < BLOCK_SIZE; i++)
		for (int j = 0; j < BLOCK_SIZE; j++)
			result(i, j) = 0;

	for (int i = 0; i < vec.size(); i++) {
		result(row,col) = vec.at(i);

		if ((row + col) % 2 == 0) { // Moving up
			if (col == 7) {
				row++;
			}
			else if (row == 0) {
				col++;
			}
			else {
				row--;
				col++;
			}
		}
		else { // Moving down
			if (row == 7) {
				col++;
			}
			else if (col == 0) {
				row++;
			}
			else {
				row++;
				col--;
			}
		}
	}
	return result;
}
Mat_<float> inverseCosine(Mat_<int> block)
{
	Mat_<float> block2(block.rows, block.cols);
	float ci, cj, aux, sum;
	for (int u = 0; u < block.rows; u++)
		for (int v = 0; v < block.cols; v++)
		{
			sum = 0;
			for (int i = 0; i < block.rows; i++)
				for (int j = 0; j < block.cols; j++)
				{
					if (i == 0)
						ci = 1/sqrt(8);
					else ci = sqrt(2)/sqrt(8);

					if (j == 0)
						cj = 1/sqrt(8);
					else cj = sqrt(2)/sqrt(8);
					aux = ci*cj*block(i, j) * cos((2 * v + 1) * i * PI / (2 * block.rows)) *
						cos((2 * u + 1) * j * PI / (2 * block.cols));
					sum += aux;
				}
			block2(u, v) = sum;
		}

	return block2;
}
Mat jpegDecompression(vector<int> vec,int height, int width)
{
	Mat img;
	vector<Mat> newChannels;
	Mat_<uchar> channel1(height, width);
	Mat_<uchar> channel2(height, width);
	Mat_<uchar> channel3(height, width);
	newChannels.push_back(channel3);
	newChannels.push_back(channel2);
	newChannels.push_back(channel1);
	int k = 0;
	for (int i=0;i<height;i+=8)
		for (int j = 0; j < width; j += 8)
		{
			for (int channel = 0; channel < 3; channel++)
			{
				vector<int> values;
				
				int x = vec.at(k);
				while (x != INT_MAX)
				{
					k++;
					values.push_back(x);
					x = vec.at(k);
				}
				k++;
				Mat_<int> result = createMatrix(values);
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
					{
						float sum = result(u, v) * (float)dataLuminance[u][v];
						result(u, v) = sum;
					}
				Mat_<float> block2 = inverseCosine(result);
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
						block2(u, v) += 128.0;
				Mat_<uchar> res(BLOCK_SIZE, BLOCK_SIZE);
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
						res(u, v) = block2(u, v);
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
					{
						int n = i + u;
						int m = j + v;
						if (isInside(channel1,n,m))
							newChannels[channel].at<uchar>(n, m) = res(u, v);
					}
			}
		}
	merge(newChannels, img);
	return img;
}
vector<int> jpegCompression(Mat img)
{
	Mat imgYCrCb;
	cvtColor(img, imgYCrCb, COLOR_BGR2YCrCb);
	vector<Mat> channels;
	vector<Mat> newChannels;
	vector<int> vec2;
	split(imgYCrCb, channels);
	int x, y;
	for (int i = 0; i < imgYCrCb.rows; i += BLOCK_SIZE) {
		for (int j = 0; j < imgYCrCb.cols; j += BLOCK_SIZE) {
			// For each plane
			split(imgYCrCb, channels);
			for (int channel = 0; channel < channels.size(); channel++) {
				// Creating a block
				Mat_<float> block(BLOCK_SIZE, BLOCK_SIZE);
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
					{
						x = i + u;
						y = j + v;
						if (isInside(imgYCrCb, x, y))
						{
							block(u, v) = channels[channel].at<uchar>(x, y);
						}
						else block(u, v) = 0;
					}
				// Subtracting the block by 128
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
					{
						block(u, v) -= 128.0;
					}
				Mat_<float> block2 = discreteCosine(block);
				transpose(block2,block2);
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
					{
						float sum = block2(u, v) / (float) dataLuminance[u][v];
						block2(u, v) = sum;
						block2(u,v)=round(block2(u, v));
					}
				Mat_<int> block3(BLOCK_SIZE, BLOCK_SIZE);
				for (int u = 0; u < BLOCK_SIZE; u++)
					for (int v = 0; v < BLOCK_SIZE; v++)
					{
						block3(u, v) = block2(u, v);
					}
				vector<int> vec=zigZagTraversal(block3);
				vector<int> vec3=eliminateZeros(vec);
				for (int u = 0; u < vec3.size(); u++)
					vec2.push_back(vec3.at(u));
				vec2.push_back(INT_MAX);
			}
		}
	}
	return vec2;
}
int main()
{
	Mat_<Vec3b> imgBGR = imread("Images/colours.bmp");
	vector<int> vec = jpegCompression(imgBGR);
	Mat result = jpegDecompression(vec, imgBGR.rows, imgBGR.cols);
	Mat dst;
	imshow("img", imgBGR);
	cvtColor(result, dst, COLOR_YCrCb2BGR);
	imshow("result", dst);
	imwrite("Images/compressed.jpg", dst);
	waitKey(0);
	destroyWindow("MyWindow");
	waitKey(0);
	return 0;
}
