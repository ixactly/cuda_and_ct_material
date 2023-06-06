//
// Created by 森智希 on 2023/04/28.
//

#ifndef IMAGEPROCESSING_IMAGE_H
#define IMAGEPROCESSING_IMAGE_H

#include <memory>
#include <functional>
#include <cstring>
#include <fstream>
#include <iostream>

template<typename T>
class Image {
public:
    Image() {};
    Image(int width, int height) : cols_(width), rows_(height),
                                   data_(new T[width * height]) {};
    Image(int width, int height, std::string &filename) : cols_(width), rows_(height) {
        read(filename, cols_, rows_);
    };

    Image(Image &&other) noexcept: rows_(other.rows_), cols_(other.cols_),
                                   data_(std::move(other.data_)) {};

    Image &operator=(Image &&other) noexcept {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
        }
        return *this;
    };

    // deep-copy
    Image(const Image &other) : rows_(other.rows_), cols_(other.cols_) {
        data_ = std::make_unique<T[]>(rows_ * cols_);
        std::memcpy(data_.get(), other.data_.get(), sizeof(T) * rows_ * cols_);
    };

    Image &operator=(const Image &other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::make_unique<T[]>(rows_ * cols_);
            std::memcpy(data_.get(), other.data_.get(), sizeof(T) * rows_ * cols_);
        }
        return *this;
    };

    ~Image() = default;

    void read(const std::string &filename, int width, int height) {
        cols_ = width;
        rows_ = height;

        data_.reset();
        data_ = std::make_unique<T[]>(cols_ * rows_);
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            std::cout << "file not read correctly. please check file path." << std::endl;
            return;
        }
        ifs.read(reinterpret_cast<char *>(data_.get()), sizeof(T) * cols_ * rows_);
    }

    void save(const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            std::cout << "file not saved correctly. please check file path." << std::endl;
        }
        ofs.write(reinterpret_cast<char *>(data_.get()), sizeof(T) * cols_ * rows_);
    }

    T &operator()(int x, int y) {
        return data_[x + y * cols_];
    }
    const T &operator()(int x, int y) const {
        return data_[x + y * cols_];
    }

    int rows() const {
        return rows_;
    }

    int cols() const {
        return cols_;
    }

    float co_x(int x, float pixel_size) const {
        return pixel_size * ((float) x - ((float) cols_ - 1.0f) / 2.0f);
    }

    float co_y(int y, float pixel_size) const {
        return pixel_size * ((float) y - ((float) rows_ - 1.0f) / 2.0f);
    }

    void reset() {
        data_.reset();
    }

    void forEach(std::function<T> f) {
        for (int y = 0; y < rows_; y++) {
            for (int x = 0; x < cols_; x++) {
                (*this)(x, y) = f((*this)(x, y));
            }
        }
    }

    void setZeros() {
        for (int y = 0; y < rows_; y++) {
            for (int x = 0; x < cols_; x++) {
                (*this)(x, y) = (T) 0;
            }
        }
    }

    void setOnes() {
        for (int y = 0; y < rows_; y++) {
            for (int x = 0; x < cols_; x++) {
                (*this)(x, y) = (T) 1;
            }
        }
    }

private:
    int rows_;
    int cols_;
    std::unique_ptr<T[]> data_;
};

#endif
