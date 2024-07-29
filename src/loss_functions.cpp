#include "loss_functions.h"
#include <cmath>

namespace loss {

float mse(const Tensor& predictions, const Tensor& targets) {
    float sum = 0.0f;
    int size = predictions.shape()[0] * predictions.shape()[1];
    for (int i = 0; i < size; ++i) {
        float diff = predictions.data()[i] - targets.data()[i];
        sum += diff * diff;
    }
    return sum / size;
}

Tensor mse_gradient(const Tensor& predictions, const Tensor& targets) {
    Tensor gradient(predictions.shape());
    int size = predictions.shape()[0] * predictions.shape()[1];
    for (int i = 0; i < size; ++i) {
        gradient.data()[i] = 2 * (predictions.data()[i] - targets.data()[i]) / size;
    }
    return gradient;
}

}