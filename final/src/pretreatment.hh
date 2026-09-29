#pragma once

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

// Configuration
struct HOGConfig
{
    cv::Size cellSize = { 8, 8 };
    cv::Size blockSize = { 16, 16 };
    cv::Size blockStride = { 8, 8 };
    int orientations = 9;
};

struct PyramidConfig
{
    float scale = 0.75f; // scale factor between levels
    int levels = 3; // number of pyramid levels
    cv::Size baseSize = { 128, 128 }; // base (largest) window size
    cv::Size cellSize = { 8, 8 };
};

struct ColorConfig
{
    int bins = 32; // bins per HSV channel
};

// Image Pyramid
// Returns a list of images at decreasing scales
// Each level is scaled by pyramidConfig.scale relative to the previous
std::vector<cv::Mat> buildPyramid(const cv::Mat& image,
                                  const PyramidConfig& cfg);

// HOG Extraction
std::vector<float> extractHOG(const cv::Mat& image, const HOGConfig& hogCfg);

// Color Histogram (HSV)
std::vector<float> extractColorHistogram(const cv::Mat& image,
                                         const ColorConfig& colorCfg);

// Full Feature Extraction: HOG pyramid + Color Histogram
// For each pyramid level:
//   - extract HOG on the resized image (upsampled back to baseSize for
//   consistent grid)
//   - weight by level (coarser levels contribute less)
// Then append global color histogram
std::vector<float> extractFeatures(const cv::Mat& image,
                                   const HOGConfig& hogCfg = {},
                                   const PyramidConfig& pyrCfg = {},
                                   const ColorConfig& colorCfg = {});
