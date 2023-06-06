#include "Image.h"
#include "Params.h"
#include "sirt.h"
#include <chrono>

int main() {
    Geometry geom = {500.0, 1000.0, 1.0, 1.0, 256, 256, 512, 360}; // see Geometry in "Params.h"
    Image<float> img(geom.img_width, geom.img_height); // reconstruct image
    Image<float> proj_measured, proj_pred(geom.detect_width, geom.detect_proj); // sinogram

    std::string input_file_name = "../../raw/lenna_proj_float_512x360.raw";
    proj_measured.read(input_file_name, geom.detect_width, geom.detect_proj);

    float alpha = 5e-7;
    int iter = 60;
    img.setOnes();

    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now();

    SIRT(img, proj_measured, geom, alpha, iter);

    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "time: " << elapsed / 1000.0 << "s." << std::endl;

    img.save("../../raw/lenna_recon_float_256x256.raw");
}