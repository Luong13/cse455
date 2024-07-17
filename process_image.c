#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int c, int h, int w)
{
    // TODO Fill this in
    if (w < 0) {
        w = 0;
    } else if (w >= im.w) {
        w = im.w - 1;
    }
    if (h < 0) {
        h = 0;
    } else if (h >= im.h) {
        h = im.h - 1;
    }
    return im.data[c*im.h*im.w + h*im.w + w];
}

void set_pixel(image im, int c, int h, int w, float v)
{
    // TODO Fill this in
    if (0 <= c && c < im.c && 0 <= h && h < im.h && 0 <= w && w < im.w) {
        im.data[c*im.h*im.w + h*im.w + w] = v;
    }
}

image copy_image(image im)
{
    image copy = make_image(im.c, im.h, im.w);
    // TODO Fill this in
    memcpy(copy.data, im.data, (im.c*im.h*im.w)*sizeof(float));
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(1, im.h, im.w);
    // TODO Fill this in
    float Y[] = {0.299, 0.587, 0.113};
    for (int i = 0; i < im.h*im.w; i++) {
        for (int j = 0; j < im.c; j++) {
            gray.data[i] += Y[j] * im.data[i + j*im.h*im.w];
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    for (int i = 0; i < im.h*im.w; i++) {
        im.data[c*im.h*im.w + i] += v;
    }
}

void clamp_image(image im)
{
    // TODO Fill this in
    for (int i = 0; i < im.c*im.h*im.w; i++) {
        if (im.data[i] < 0.0) {
            im.data[i] = 0.0;
        } else if (im.data[i] > 1.0) {
            im.data[i] = 1.0;
        }
    }
}

// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    float r,g,b,h,s,v,m,c;
    for (int i = 0; i < im.h*im.w; i++) {
        r = im.data[i];
        g = im.data[i + im.h*im.w];
        b = im.data[i + 2*im.h*im.w];
        v = three_way_max(r,g,b);
        m = three_way_min(r,g,b);
        c = v-m;
        if (v == 0) {
            s = 0;
        } else {
            s = c/v;
        }
        if (c == 0) {
            h = 0;
        } else if (v == r) {
            h = (g-b)/c;
        } else if (v == g) {
            h = (b-r)/c + 2;
        } else {
            h = (r-g)/c + 4;
        }
        if (h < 0) {
            h = h/6 + 1;
        } else {
            h = h/6;
        }
        im.data[i] = h;
        im.data[i + im.h*im.w] = s;
        im.data[i + 2*im.h*im.w] = v;
    }
}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    float r,g,b,h,s,v,m,c;
    for (int i = 0; i < im.h*im.w; i++) {
        h = im.data[i];
        s = im.data[i + im.h*im.w];
        v = im.data[i + 2*im.h*im.w];
        h = h*6;
        c = v*s;
        r = g = b = m = v - c;
        float x = c*(1.0 - fabs(fmod(h,2.0) - 1.0));
        if (0 <= h && h < 1) {
            r += c;
            g += x;
            b += 0;
        } else if (1 <= h && h < 2) {
            r += x;
            g += c;
            b += 0;
        } else if (2 <= h && h < 3) {
            r += 0;
            g += c;
            b += x;
        } else if (3 <= h && h < 4) {
            r += 0;
            g += x;
            b += c;
        } else if (4 <= h && h < 5) {
            r += x;
            g += 0;
            b += c;
        } else if (5 <= h && h < 6) {
            r += c;
            g += 0;
            b += x;
        }

        im.data[i] = r;
        im.data[i + im.h*im.w] = g;
        im.data[i + 2*im.h*im.w] = b;
    }
}

void scale_image(image im, int c, float v) {
    for (int i = 0; i < im.h*im.w; i++) {
        im.data[c*im.h*im.w + i] *= v;
    }
}
