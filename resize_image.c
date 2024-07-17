#include <math.h>
#include "image.h"

float nn_interpolate(image im, int c, float h, float w)
{
    // TODO
    int x = round(w);
    int y = round(h);
    return get_pixel(im, c, y, x);
}

image nn_resize(image im, int h, int w)
{
    // TODO
    image resized_im = make_image(im.c, h, w);
    float scale_x = (float)im.w / w;
    float scale_y = (float)im.h / h;
    for (int i = 0; i < resized_im.c; i++) {
        for (int j = 0; j < resized_im.h; j++) {
            for (int k = 0; k < resized_im.w; k++) {
                float x = -0.5 + scale_x/2.0 + k * scale_x;
                float y = -0.5 + scale_y/2.0 + j * scale_y;
                set_pixel(resized_im, i, j, k, nn_interpolate(im, i, y, x));
            }
        }
    }
    return resized_im;
}

float bilinear_interpolate(image im, int c, float h, float w)
{
    // TODO
    int left, right, top, bottom;
    left = floor(w);
    right = ceil(w);
    top = ceil(h);
    bottom = floor(h);

    float V1, V2, V3, V4;
    V1 = get_pixel(im, c, top, left);
    V2 = get_pixel(im, c, top, right);
    V3 = get_pixel(im, c, bottom, left);
    V4 = get_pixel(im, c, bottom, right);

    float d1, d2, d3, d4;
    d1 = w - left;
    d2 = right - w;
    d3 = top - h;
    d4 = h - bottom;

    float q, q1, q2;
    q1 = V1*d2 + V2*d1;
    q2 = V3*d2 + V4*d1;
    q = q1*d4 + q2*d3;

    return q;
}

image bilinear_resize(image im, int h, int w)
{
    // TODO
    image resized_im = make_image(im.c, h, w);
    float scale_x = (float)im.w / w;
    float scale_y = (float)im.h / h;
    for (int i = 0; i < resized_im.c; i++) {
        for (int j = 0; j < resized_im.h; j++) {
            for (int k = 0; k < resized_im.w; k++) {
                float x = -0.5 + scale_x/2.0 + k * scale_x;
                float y = -0.5 + scale_y/2.0 + j * scale_y;
                set_pixel(resized_im, i, j, k, bilinear_interpolate(im, i, y, x));
            }
        }
    }
    return resized_im;
}

