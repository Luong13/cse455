#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
    // TODO
    float sum = 0.0;
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                sum += get_pixel(im, i, j, k);
            }
        }
    }
    if (sum == 0) {
        return;
    }
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                set_pixel(im, i, j, k, get_pixel(im, i, j, k)/sum);
            }
        }
    }
}

image make_box_filter(int w)
{
    // TODO
    image box_filter = make_image(1, w, w);
    float num_pixels = (float)w * (float)w;
    float q = (float)1.0 / num_pixels;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            set_pixel(box_filter, 0, i, j, q);
        }
    }
    return box_filter;
}

void convolve_channel(image im, image filter, int im_c, int fil_c, image* result)
{
    int x_half = filter.w / 2;
    int y_half = filter.h / 2;
    for (int i = 0; i < im.h; i++) {
        for (int j = 0; j < im.w; j++) {
            float q = 0.0;
            for (int y = 0; y < filter.h; y++) {
                for (int x = 0; x < filter.w; x++) {
                    int im_x_pix = j - x_half + x;
                    int im_y_pix = i - y_half + y;
                    q += get_pixel(filter, fil_c, y, x) * get_pixel(im, im_c, im_y_pix, im_x_pix);
                }
            }
            set_pixel(*result, im_c, i, j, q);
        }
    }
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    assert(filter.c == im.c || filter.c == 1);
    image new_im = make_image(im.c, im.h, im.w);
    if (filter.c == im.c) {
        for (int i = 0; i < im.c; i++) {
            convolve_channel(im, filter, i, i, &new_im);
        }
    } else {
        for (int i = 0; i < im.c; i++) {
            convolve_channel(im, filter, i, 0, &new_im);
        }
    }

    if (preserve != 1) {
        //sum the stuff
        image sum_im = make_image(1, im.h, im.w);
        for (int i = 0; i < im.h; i++) {
            for (int j = 0; j < im.w; j++) {
                float q = 0.0;
                for (int k = 0; k < im.c; k++) {
                    q += get_pixel(new_im, k, i, j);
                }
                set_pixel(sum_im, 0, i, j, q);
            }
        }
        return sum_im;
    }
    return new_im;
}

image make_highpass_filter()
{
    // TODO
    image filter = make_image(1,3,3);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 0, 0, 1, -1);
    set_pixel(filter, 0, 0, 2, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 0, 1, 1, 4);
    set_pixel(filter, 0, 1, 2, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 0, 2, 1, -1);
    set_pixel(filter, 0, 2, 2, 0);
    return filter;
}

image make_sharpen_filter()
{
    // TODO
    image filter = make_image(1,3,3);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 0, 0, 1, -1);
    set_pixel(filter, 0, 0, 2, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 0, 1, 1, 5);
    set_pixel(filter, 0, 1, 2, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 0, 2, 1, -1);
    set_pixel(filter, 0, 2, 2, 0);
    return filter;
}

image make_emboss_filter()
{
    // TODO
    image filter = make_image(1,3,3);
    set_pixel(filter, 0, 0, 0, -2);
    set_pixel(filter, 0, 0, 1, -1);
    set_pixel(filter, 0, 0, 2, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 0, 1, 1, 1);
    set_pixel(filter, 0, 1, 2, 1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 0, 2, 1, 1);
    set_pixel(filter, 0, 2, 2, 2);
    return filter;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: We should NOT use preserve for the highpass filter because it is only looking for edges and so the output
// does not need to worry about RGB values. We do need to use preserve for the sharpen and emboss filters because the output
// does retain colors/RGB values

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: We have to do post-processing for all of them because we need to normalize after applying
// any of those filters to make sure the values of the pixels stay within the valid range

image make_gaussian_filter(float sigma)
{
    // TODO
    int size = 6 * sigma;
    if (size % 2 == 0) {
        size += 1;
    }
    image filter = make_image(1, size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = j - (size-1)/2;
            float y = i - (size-1)/2;
            float q = (1.0/(TWOPI*powf(sigma,2.0)) * expf(-(powf(x,2.0)+powf(y,2.0))/(2*powf(sigma,2.0))));
            set_pixel(filter, 0, i, j, q);
        }
    }
    l1_normalize(filter);
    return filter;
}

image add_image(image a, image b)
{
    // TODO
    assert(a.c == b.c && a.h == b.h && a.w == b.w);
    image result = make_image(a.c, a.h, a.w);
    for (int i = 0; i < a.c; i++) {
        for (int j = 0; j < a.h; j++) {
            for (int k = 0; k < a.w; k++) {
                set_pixel(result, i, j, k, get_pixel(a, i, j, k) + get_pixel(b, i, j, k));
            }
        }
    }
    return result;
}

image sub_image(image a, image b)
{
    // TODO
    assert(a.c == b.c && a.h == b.h && a.w == b.w);
    image result = make_image(a.c, a.h, a.w);
    for (int i = 0; i < a.c; i++) {
        for (int j = 0; j < a.h; j++) {
            for (int k = 0; k < a.w; k++) {
                set_pixel(result, i, j, k, get_pixel(a, i, j, k) - get_pixel(b, i, j, k));
            }
        }
    }
    return result;
}

image make_gx_filter()
{
    // TODO
    image filter = make_image(1,3,3);
    set_pixel(filter, 0, 0, 0, -1);
    set_pixel(filter, 0, 0, 1, 0);
    set_pixel(filter, 0, 0, 2, 1);
    set_pixel(filter, 0, 1, 0, -2);
    set_pixel(filter, 0, 1, 1, 0);
    set_pixel(filter, 0, 1, 2, 2);
    set_pixel(filter, 0, 2, 0, -1);
    set_pixel(filter, 0, 2, 1, 0);
    set_pixel(filter, 0, 2, 2, 1);
    return filter;
}

image make_gy_filter()
{
    // TODO
    image filter = make_image(1,3,3);
    set_pixel(filter, 0, 0, 0, -1);
    set_pixel(filter, 0, 0, 1, -2);
    set_pixel(filter, 0, 0, 2, -1);
    set_pixel(filter, 0, 1, 0, 0);
    set_pixel(filter, 0, 1, 1, 0);
    set_pixel(filter, 0, 1, 2, 0);
    set_pixel(filter, 0, 2, 0, 1);
    set_pixel(filter, 0, 2, 1, 2);
    set_pixel(filter, 0, 2, 2, 1);
    return filter;
}

void feature_normalize(image im)
{
    // TODO
    float max = get_pixel(im, 0, 0, 0);
    float min = get_pixel(im, 0, 0, 0);
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                float q = get_pixel(im, i, j, k);
                if (q > max) {
                    max = q;
                }
                if (q < min) {
                    min = q;
                }
            }
        }
    }
    float range = max - min;
    
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                if (range == 0) {
                    set_pixel(im, i, j, k, 0);
                } else {
                    set_pixel(im, i, j, k, (get_pixel(im, i, j, k) - min)/range);
                }
            }
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    image* result = calloc(2, sizeof(image));
    image gx = convolve_image(im, make_gx_filter(), 0);
    image gy = convolve_image(im, make_gy_filter(), 0);
    image mag_im = make_image(1, im.h, im.w);
    image dir_im = make_image(1, im.h, im.w);

    for (int j = 0; j < im.h; j++) {
        for (int k = 0; k < im.w; k++) {
            float gx_pix = get_pixel(gx, 0, j, k);
            float gy_pix = get_pixel(gy, 0, j, k);
            set_pixel(mag_im, 0, j, k, sqrtf(gx_pix*gx_pix + gy_pix*gy_pix));
            set_pixel(dir_im, 0, j, k, atan2f(gy_pix,gx_pix));
        }
    }
    result[0] = mag_im;
    result[1] = dir_im;
    return result;
}

image colorize_sobel(image im)
{
    // TODO
    image* sobel = sobel_image(im);
    feature_normalize(sobel[0]);
    feature_normalize(sobel[1]);

    image result = make_image(im.c, im.h, im.w);

    for (int i = 0; i < im.h; i++) {
        for (int j = 0; j < im.w; j++) {
            set_pixel(result, 0, i, j, get_pixel(sobel[1], 0, i, j));
            set_pixel(result, 1, i, j, get_pixel(sobel[0], 0, i, j));
            set_pixel(result, 2, i, j, get_pixel(sobel[0], 0, i, j));
        }
    }


    return result;
}
