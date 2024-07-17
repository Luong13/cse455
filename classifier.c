#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                // TODO
                m.data[i][j] = 1 / (1 + exp(-x));
            } else if (a == RELU){
                // TODO
                m.data[i][j] = MAX(x, 0);
            } else if (a == LRELU){
                // TODO
                m.data[i][j] = MAX(x, 0.1*x); 
            } else if (a == SOFTMAX){
                // TODO
                m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
            for (int j = 0; j < m.cols; j++) {
                m.data[i][j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            double grad = 0.0;
            if(a == LOGISTIC){
                // TODO
                grad = x * (1 - x);
            } else if (a == RELU){
                // TODO
                grad = (x > 0);
            } else if (a == LRELU){
                // TODO
                grad = MAX(x > 0, 0.1); 
            } else if (a == SOFTMAX){
                // TODO
                grad = 1;
            }
            d.data[i][j] *= grad;
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // TODO: fix this! multiply input by weights and apply activation function.
    matrix out = matrix_mult_matrix(in, l->w);
    activate_matrix(out, l->activation);

    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
    gradient_matrix(l->out, l->activation, delta);


    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix dw = matrix_mult_matrix(transpose_matrix(l->in), delta); // replace this
    l->dw = dw;

    
    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    matrix dx = matrix_mult_matrix(delta, transpose_matrix(l->w)); // replace this

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    matrix dw = axpy_matrix(momentum, l->v, axpy_matrix(-decay, l->w, l->dw));
    free_matrix(l->v);
    l->v = dw;

    // Update l->w
    dw = axpy_matrix(rate, l->v, l->w);
    free_matrix(l->w);
    l->w = dw;

    // Remember to free any intermediate results to avoid memory leaks

}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions 
//
// 5.2.2.1 Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
// TODO
// Training accuracy tells us how well our model fits the training data and testing accuracy tells us how well our model fits the test data (data our model hasn't seen before).
// By knowing both accuracies we can sort of tell how well our model will do with predicting future unknown data and whether or not our model is overfitting our training data
//
// 5.2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// TODO
// Rate         Train       Test        Loss
// 10           0.0987      0.098       -nan
// 1            0.8844      0.8775      Varies 0.4-0.8
// 0.1          0.9171      0.9177      Varies 0.1-0.3
// 0.01         0.90385     0.909       Varies 0.3-0.5
// 0.001        0.8587      0.8678      Varies 0.5-0.7
// It seems that there is an optimal value for the learning rate at somewhere around 0.1. This rate had the lowest loss as well as the highest accuracy. The further the learning rate value is from this optimal value, the higher loss and lower accuracy the model has
//
// 5.2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// TODO
// Decay        Train       Test
// 1            0.8824      0.8902
// 0.1          0.9119      0.9137
// 0.01         0.9166      0.9176
// 0.001        0.9171      0.9176
// 0.0001       0.9170      0.9176
// 0.00001      0.9171      0.9177
// It seems that using any values at and below 0.001 for weight decay results in pretty much the same accuracy for both traing and test. But weight decay values above 0.001 results in lower accuracy for both train and test
//
// 5.2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// TODO
// Training using the following parameters: batch=128, iters=1000, rate=.01, momentum=.9, decay=.0
// Activation   Train       Test
// LOGISTIC     0.8884      0.8916
// RELU         0.9267      0.9277
// LRELU        0.9240      0.9257
// With the default parameters, RELU activation had the best accuracy for both train and test
//
// 5.2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// TODO
// Training using the following parameters: batch=128, iters=1000, momentum=.9, decay=.0, activation=RELU-SOFTMAX
// Rate         Train       Test
// 10           0.0987      0.098
// 1            0.0987      0.098
// 0.1          0.9603      0.9543
// 0.01         0.9267      0.9277
// 0.001        0.8645      0.8698
// The best learning rate should be somewhere around 0.1   
//
// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// TODO
// Training using the following parameters: batch=128, iters=1000, rate=.1, momentum=.9, activation=RELU-SOFTMAX
// Decay        Train       Test
// 1            0.9129      0.9141
// 0.1          0.9599      0.9552
// 0.01         0.9612      0.9535
// 0.001        0.9603      0.9545
// 0.0001       0.9623      0.9546
// 0.00001      0.9606      0.954
// Having a small decay helps both train and test accuracies very very slightly. Andd having large decay hurts both accuracies but also makes the two accuracies closer to being equal.
// This is because the decay is a regularization so larger values of decay lowers variance (less difference between train and test accuracies) while also increasing bias (lower accuracies for both train and test)
//
// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// TODO
// Training using the following parameters: batch=128, iters=3000, rate=.1, momentum=.9, activation=RELU-RELU-SOFTMAX
// Decay        Train       Test
// 1            0.9303      0.9314
// 0.1          0.9744      0.966
// 0.01         0.9727      0.9597
// 0.001        0.9749      0.9631
// 0.0001       0.9826      0.9671
// 0            0.9823      0.9673
// It seems that the best decay value is little to no decay at all. This may be because using an extra layer and more iterations lowers the risk of overfitting so
// so adding in significant decay will only hurt the bias without improving the variance very much
// 
// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// TODO
// Trained using the following parameters: batch=128, iters=3000, rate=.01, momentum=.9, decay=.01, activation=RELU-RELU-SOFTMAX
// Train accuracy: 0.45982
// Test accuracy: 0.4413



