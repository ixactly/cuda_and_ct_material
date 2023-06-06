//
// Created by 森智希 on 2023/06/05.
//

#ifndef RECONSTRUCTION_PARAMS_H
#define RECONSTRUCTION_PARAMS_H

#define PI 3.141592

struct Geometry {
    float sod; // source_object_distance
    float sdd; // source_detector_distance
    float detector_pixel_size;
    float image_pixel_size;
    int img_width;
    int img_height;
    int detect_width;
    int detect_proj;
};
#endif //RECONSTRUCTION_PARAMS_H
