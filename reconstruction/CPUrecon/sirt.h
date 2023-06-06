#include "Image.h"
#include "Params.h"
#include <cmath>

inline void forward(Image<float> &sinogram, const Image<float> &img, const Geometry &geom) {
    float x, y, t, theta = 0.0f;
    float pix_size = geom.image_pixel_size, det_size = geom.detector_pixel_size;
    const int n_proj = geom.detect_proj; // 投影角の数
    float d_theta = 2.0f * PI / (float) n_proj;

    // 順投影によるsinogramの生成
    // 投影角ごとに順投影計算を行い、sinogramに線積分値を加算する
    for (int k = 0; k < n_proj; k++) {
        for (int i = 0; i < geom.img_width; i++) {
            for (int j = 0; j < geom.img_height; j++) {
                x = img.co_x(i, pix_size);
                y = img.co_y(j, pix_size);

                // 再構成画像の(i, j)インデックスの座標から、検出器のインデックスを求める
                t = ((x * std::cos(theta) - y * std::sin(theta) + det_size *
                        (float) (geom.detect_width - 1) / 2.0f)) / det_size;
                int t_idx = std::floor(t);
                if (t_idx < 0 || geom.detect_width - 2 < t_idx) continue;
                sinogram(t_idx, k) += img(i, j) * pix_size * ((float) t_idx + 1.0f - t);
                sinogram(t_idx + 1, k) += img(i, j) * pix_size * (t - (float) t_idx);

            }
        }
        theta += d_theta;
    }
}
inline void calc_error(Image<float>& err_sino, const Image<float>& pred, const Image<float>& measured, Geometry& geom) {
    for (int i = 0; i < geom.detect_width; i++) {
        for (int j = 0; j < geom.detect_proj; j++) {
            err_sino(i, j) = (measured(i, j) - pred(i, j));
        }
    }
}

inline void backward(Image<float>& img, const Image<float>& err_sino, const Geometry& geom) {
    float x, y, t, theta = 0.0f;
    float pix_size = geom.image_pixel_size, det_size = geom.detector_pixel_size;
    const int n_proj = geom.detect_proj; // 投影角の数
    float d_theta = 2.0f * PI / (float) n_proj;

    // 逆投影によるimageの生成
    // 投影角ごとに逆投影計算を行う
    for (int k = 0; k < n_proj; k++) {
        for (int i = 0; i < geom.img_width; i++) {
            for (int j = 0; j < geom.img_height; j++) {
                x = img.co_x(i, pix_size);
                y = img.co_y(j, pix_size);

                // 再構成画像の(i, j)インデックスの座標から、検出器のインデックスを求める
                t = ((x * std::cos(theta) - y * std::sin(theta) + det_size *
                        (float) (geom.detect_width - 1) / 2.0f)) / det_size;
                int t_idx = std::floor(t);
                if (t_idx < 0 || geom.detect_width - 2 < t_idx) continue;
                img(i, j) += err_sino(t_idx, k) * pix_size * ((float) t_idx + 1.0f - t)
                             + pix_size * (t - (float) t_idx) * err_sino(t_idx + 1, k);
            }
        }
        theta += d_theta;
    }
}

inline void feedback(Image<float>& img, const Image<float>& err_img, Geometry& geom, float alpha) {
    for (int i = 0; i < geom.img_width; i++) {
        for (int j = 0; j < geom.img_height; j++) {
            img(i, j) += alpha * err_img(i, j);
        }
    }
}

inline void SIRT(Image<float>& img, const Image<float>& sinogram, Geometry& geom, float alpha, int iter) {
    Image<float> pred(geom.detect_width, geom.detect_proj);
    Image<float> err_sino(geom.detect_width, geom.detect_proj);
    Image<float> err_img(geom.img_width, geom.img_height);

    for (int i = 0; i < iter; i++) {
        pred.setZeros();
        std::cout << "iteration: " << i + 1 << std::endl;

        forward(pred, img, geom);
        calc_error(err_sino, pred, sinogram, geom);
        backward(err_img, err_sino, geom);
        feedback(img, err_img, geom, alpha);
    }
}
