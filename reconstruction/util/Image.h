//
// Created by 森智希 on 2023/04/28.
//

#ifndef IMAGEPROCESSING_IMAGE_H
#define IMAGEPROCESSING_IMAGE_H

#include <memory>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "params.h"

namespace bip {
    template<typename T>
    class Image {
    public:
        Image(int height, int width, int channel) : rows_(height), cols_(width), dims_(channel),
                                                    data_(new T[width * height * channel]) {};
        Image(Image &&other) noexcept: rows_(other.rows_), cols_(other.cols_), dims_(other.dims_),
                                       data_(std::move(other.data_)) {};
        Image &operator=(Image &&other) noexcept {
            if (this != &other) {
                rows_ = other.rows_;
                cols_ = other.cols_;
                dims_ = other.dims_;
                data_ = std::move(other.data_);
            }
            return *this;
        };

        // copiable
        Image(const Image &) = delete;
        Image &operator=(const Image &) = delete;

        ~Image() = default;

        T &operator()(int x, int y, int c) {
            return data_[c + x * dims_ + y * rows_ * dims_];
        }

        const T &operator()(int x, int y, int c) const {
            return data_[c + x * dims_ + y * rows_ * dims_];
        }

        int rows() const {
            return rows_;
        }
        int cols() const {
            return cols_;
        }
        int dims() const {
            return dims_;
        }

        void reset() {
            data_.reset();
        }

        void forEach(std::function<T> f) {
            for (int c = 0; c < dims_; c++) {
                for (int y = 0; y < cols_; y++) {
                    for (int x = 0; x < rows_; x++) {
                        this(x, y, c) = f(this(x, y, c));
                    }
                }
            }
        }

    private:
        int rows_;
        int cols_;
        int dims_;
        std::unique_ptr<T[]> data_;
    };

    template<typename T>
    Image<T> imread(const std::string &path) {
        cv::Mat cv_img = cv::imread(path, cv::IMREAD_UNCHANGED);

        // 画像形式によって処理を分ける
        int step = (int) cv_img.step;

        // 画像の幅と高さを取得する
        int width = cv_img.cols;
        int height = cv_img.rows;
        int dims = cv_img.channels();
        // Image型
        Image<T> img(width, height, dims);

        if (dims == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    img(x, y, 0) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize()]);
                    // std::cout << sizeof(img(x, y, 0)) << std::endl;
                }
            }
        } else if (dims == 3) {

            // 各チャンネルのrawデータを取得する
            // ここがT型ではなく, U型になる．U型は読み込むcv::Mat型のdata型に依存する

            // rawデータを出力する
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    img(x, y, 0) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize() + 2]); // Rチャンネル
                    img(x, y, 1) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize() + 1]); // Gチャンネル
                    img(x, y, 2) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize() + 0]); // Bチャンネル
                }
            }
        } else if (dims == 4) {
            // RGBA画像の場合
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    img(x, y, 0) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize() + 2]); // Rチャンネル
                    img(x, y, 1) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize() + 1]); // Gチャンネル
                    img(x, y, 2) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize() + 0]); // Bチャンネル
                    img(x, y, 3) = static_cast<T>(cv_img.data[y * cv_img.step + x * cv_img.elemSize() + 3]); // Aチャンネル
                }
            }
        }
        return img;
    }

    template<typename T>
    void imwrite(const std::string &path, const Image<T> &img) {
        cv::Mat cv_img;
        int dims = img.dims();
        if (dims == 1) {
            cv_img = cv::Mat(img.cols(), img.rows(), CV_8U);
            for (int y = 0; y < img.cols(); y++) {
                for (int x = 0; x < img.rows(); x++) {
                    cv_img.at<uchar>(y, x) = static_cast<uchar>(img(x, y, 0));
                }
            }
        } else if (dims == 3) {
            cv_img = cv::Mat(img.cols(), img.rows(), CV_8UC3);
            for (int y = 0; y < img.cols(); y++) {
                for (int x = 0; x < img.rows(); x++) {
                    cv_img.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(img(x, y, 2));
                    cv_img.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(img(x, y, 1));
                    cv_img.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(img(x, y, 0));
                }
            }
        } else if (dims == 4) {
            cv_img = cv::Mat(img.cols(), img.rows(), CV_8UC4);
            for (int y = 0; y < img.cols(); y++) {
                for (int x = 0; x < img.rows(); x++) {
                    cv_img.at<cv::Vec4b>(y, x)[0] = static_cast<uchar>(img(x, y, 2));
                    cv_img.at<cv::Vec4b>(y, x)[1] = static_cast<uchar>(img(x, y, 1));
                    cv_img.at<cv::Vec4b>(y, x)[2] = static_cast<uchar>(img(x, y, 0));
                    cv_img.at<cv::Vec4b>(y, x)[3] = static_cast<uchar>(img(x, y, 3));
                }
            }
        }
        cv::imwrite(path, cv_img);
    }

    template<typename T>
    void convertToGray(Image<T> &img) {
        Image<T> gray(img.rows(), img.cols(), 1);
        for (int x = 0; x < gray.cols(); x++) {
            for (int y = 0; y < gray.rows(); y++) {
                gray(x, y, 0) = (img(x, y, 0) + img(x, y, 1) + img(x, y, 2)) / 3.0;
            }
        }
        img = std::move(gray);
    }
}
#endif //IMAGEPROCESSING_IMAGE_H
