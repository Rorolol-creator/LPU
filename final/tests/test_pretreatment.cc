#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <opencv2/opencv.hpp>

#include "../src/pretreatment.hh"

// Helpers
cv::Mat makeRGBImage(int h, int w, cv::Vec3b color) {
    return cv::Mat(h, w, CV_8UC3, color);
}

cv::Mat makeGrayImage(int h, int w, uchar value) {
    return cv::Mat(h, w, CV_8UC1, cv::Scalar(value));
}

// Image Pyramid Tests
TEST_CASE("Pyramid correct number of levels", "[pyramid]") {
    PyramidConfig cfg;
    cfg.levels   = 3;
    cfg.baseSize = {128, 128};
    cfg.scale    = 0.75f;

    cv::Mat image = makeRGBImage(256, 256, {100, 150, 200});
    auto pyramid  = buildPyramid(image, cfg);

    REQUIRE(pyramid.size() == 3);
}

TEST_CASE("Pyramid base level is baseSize", "[pyramid]") {
    PyramidConfig cfg;
    cfg.levels   = 3;
    cfg.baseSize = {128, 128};

    cv::Mat image = makeRGBImage(512, 300, {0, 0, 0});
    auto pyramid  = buildPyramid(image, cfg);

    REQUIRE(pyramid[0].cols == 128);
    REQUIRE(pyramid[0].rows == 128);
}

TEST_CASE("Pyramid levels size decrease", "[pyramid]") {
    PyramidConfig cfg;
    cfg.levels   = 3;
    cfg.baseSize = {128, 128};
    cfg.scale    = 0.75f;

    cv::Mat image = makeRGBImage(128, 128, {0, 0, 0});
    auto pyramid  = buildPyramid(image, cfg);

    for (size_t i = 1; i < pyramid.size(); i++) {
        REQUIRE(pyramid[i].cols < pyramid[i-1].cols);
        REQUIRE(pyramid[i].rows < pyramid[i-1].rows);
    }
}

// ============================================================
// HOG Tests
// ============================================================
TEST_CASE("HOG returns non-empty", "[hog]") {
    HOGConfig cfg;
    cv::Mat image = makeRGBImage(128, 128, {100, 100, 100});
    auto desc = extractHOG(image, cfg);

    REQUIRE_FALSE(desc.empty());
}

TEST_CASE("HOG descriptor size consistent", "[hog]") {
    HOGConfig cfg;
    cv::Mat img1 = makeRGBImage(128, 128, {100, 50, 200});
    cv::Mat img2 = makeRGBImage(128, 128, {10,  20, 30});

    auto desc1 = extractHOG(img1, cfg);
    auto desc2 = extractHOG(img2, cfg);

    REQUIRE(desc1.size() == desc2.size());
}

TEST_CASE("HOG works on grayscale", "[hog]") {
    HOGConfig cfg;
    cv::Mat gray = makeGrayImage(128, 128, 128);
    auto desc = extractHOG(gray, cfg);

    REQUIRE_FALSE(desc.empty());
}

// Color Histogram Tests
TEST_CASE("Color histogram correct size", "[color]") {
    ColorConfig cfg;
    cfg.bins = 32;

    cv::Mat image = makeRGBImage(128, 128, {100, 200, 50});
    auto hist = extractColorHistogram(image, cfg);

    REQUIRE(hist.size() == (size_t)(cfg.bins * 3));
}

TEST_CASE("Color histogram normalized", "[color]") {
    ColorConfig cfg;
    cfg.bins = 32;

    cv::Mat image = makeRGBImage(128, 128, {100, 200, 50});
    auto hist = extractColorHistogram(image, cfg);

    // Each channel sums to 1.0
    for (int c = 0; c < 3; c++) {
        float sum = 0.0f;
        for (int i = 0; i < cfg.bins; i++)
            sum += hist[c * cfg.bins + i];
        REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0f, 0.0f));
    }
}

TEST_CASE("Uniform color image has single non-zero bin", "[color]") {
    ColorConfig cfg;
    cfg.bins = 32;

    // Pure red image => all pixels same color => one bin should be 1.0
    cv::Mat image = makeRGBImage(128, 128, {0, 0, 255}); // BGR red
    auto hist = extractColorHistogram(image, cfg);

    int nonZero = 0;
    for (float v : hist)
        if (v > 0.0f) nonZero++;

    // Each channel should have exactly 1 non-zero bin
    REQUIRE(nonZero == 3);
}

TEST_CASE("Grayscale image returns zero color histogram", "[color]") {
    ColorConfig cfg;
    cv::Mat gray = makeGrayImage(128, 128, 100);
    auto hist = extractColorHistogram(gray, cfg);

    for (float v : hist)
        REQUIRE(v == 0.0f);
}

// Pipeline Tests
TEST_CASE("Feature vector consistent size", "[pipeline]") {
    HOGConfig     hogCfg;
    PyramidConfig pyrCfg;
    ColorConfig   colorCfg;

    cv::Mat img1 = makeRGBImage(300, 400, {100, 50, 200});
    cv::Mat img2 = makeRGBImage(128, 128, {10,  20, 30});
    cv::Mat img3 = makeRGBImage(64,  256, {255, 0,  0});

    auto f1 = extractFeatures(img1, hogCfg, pyrCfg, colorCfg);
    auto f2 = extractFeatures(img2, hogCfg, pyrCfg, colorCfg);
    auto f3 = extractFeatures(img3, hogCfg, pyrCfg, colorCfg);

    REQUIRE(f1.size() == f2.size());
    REQUIRE(f2.size() == f3.size());
}

TEST_CASE("Empty image throws exception", "[pipeline]") {
    HOGConfig     hogCfg;
    PyramidConfig pyrCfg;
    ColorConfig   colorCfg;

    cv::Mat empty;
    REQUIRE_THROWS_AS(
        extractFeatures(empty, hogCfg, pyrCfg, colorCfg),
        std::invalid_argument
    );
}
