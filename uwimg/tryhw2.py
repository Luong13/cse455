from uwimg import *
im = load_image("data/dog.jpg")
f = make_highpass_filter()
blur = convolve_image(im, f, 0)
clamp_image(blur)
save_image(blur, "highpass")