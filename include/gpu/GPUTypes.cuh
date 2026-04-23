#pragma once


//this struct replace cv::Vec3b since OpenCV types are not safe on CUDA


struct RGB8 {
    unsigned char x;
    unsigned char y;
    unsigned char z;
};
