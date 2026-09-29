#include "pretreatment.hh"

std::vector<cv::Mat> buildPyramid(const cv::Mat& image,
                                  const PyramidConfig& cfg)
{
    std::vector<cv::Mat> pyramid;

    cv::Mat current;
    cv::resize(image, current, cfg.baseSize);
    pyramid.push_back(current);

    for (int i = 1; i < cfg.levels; i++)
    {
        cv::Mat scaled;
        cv::Size newSize(static_cast<int>(current.cols * cfg.scale),
                         static_cast<int>(current.rows * cfg.scale));
        if (newSize.width < cfg.cellSize.width * 2
            || newSize.height < cfg.cellSize.height * 2)
            break; // too small
        cv::resize(current, scaled, newSize);
        pyramid.push_back(scaled);
        current = scaled;
    }

    return pyramid;
}

std::vector<float> extractHOG(const cv::Mat& image, const HOGConfig& hogCfg)
{
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    cv::HOGDescriptor hog(image.size(), hogCfg.blockSize, hogCfg.blockStride,
                          hogCfg.cellSize, hogCfg.orientations);

    std::vector<float> descriptors;
    hog.compute(gray, descriptors);
    return descriptors;
}

std::vector<float> extractColorHistogram(const cv::Mat& image,
                                         const ColorConfig& colorCfg)
{
    if (image.channels() < 3)
        // Grayscale => zero histogram
        return std::vector<float>(colorCfg.bins * 3, 0.0f);

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // H: 0-180, S: 0-255, V: 0-255
    float hRange[] = { 0, 181 };
    float svRange[] = { 0, 256 };
    const float* hRangePtr = hRange;
    const float* svRangePtr = svRange;

    std::vector<float> features;
    features.reserve(colorCfg.bins * 3);

    for (int c = 0; c < 3; c++)
    {
        cv::Mat hist;
        const float* range = (c == 0) ? hRangePtr : svRangePtr;
        cv::calcHist(&channels[c], 1, nullptr, cv::Mat(), hist, 1,
                     &colorCfg.bins, &range);

        // L1 normalize
        float sum = 0.0f;
        for (int i = 0; i < colorCfg.bins; i++)
            sum += hist.at<float>(i);
        if (sum > 0.0f)
            hist /= sum;

        for (int i = 0; i < colorCfg.bins; i++)
            features.push_back(hist.at<float>(i));
    }

    return features;
}

std::vector<float> extractFeatures(const cv::Mat& image,
                                   const HOGConfig& hogCfg,
                                   const PyramidConfig& pyrCfg,
                                   const ColorConfig& colorCfg)
{
    if (image.empty())
        throw std::invalid_argument("Empty image passed to extractFeatures");

    std::vector<float> allFeatures;

    // HOG on each level
    auto pyramid = buildPyramid(image, pyrCfg);

    for (size_t level = 0; level < pyramid.size(); level++)
    {
        const cv::Mat& levelImg = pyramid[level];

        // Resize back to base size so size stays consistent
        cv::Mat normalized;
        cv::resize(levelImg, normalized, pyrCfg.baseSize);

        auto hogFeats = extractHOG(normalized, hogCfg);
        // std::cout << "len(hog) = " << hogFeats.size() << std::endl;

        // Weight coarser levels less
        float weight = 1.0f
            / std::max(static_cast<float>(2 * level), 1.0f); // 1.0, 0.5, 0.25
        for (float& v : hogFeats)
            v *= weight;

        allFeatures.insert(allFeatures.end(), hogFeats.begin(), hogFeats.end());
    }

    // color histogram
    cv::Mat base;
    cv::resize(image, base, pyrCfg.baseSize);
    auto colorFeats = extractColorHistogram(base, colorCfg);
    allFeatures.insert(allFeatures.end(), colorFeats.begin(), colorFeats.end());

    return allFeatures;
}
