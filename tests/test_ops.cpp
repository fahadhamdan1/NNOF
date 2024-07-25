#include "ops.h"
#include "tensor.h"
#include <cassert>
#include <iostream>

void test_add_cpu() {
    auto a = std::make_shared<Tensor>(std::vector<int>{2, 2});
    auto b = std::make_shared<Tensor>(std::vector<int>{2, 2});
    auto result = std::make_shared<Tensor>(std::vector<int>{2, 2});

    // init tensors
    for (int i = 0; i < 4; ++i) {
        a->data()[i] = i;
        b->data()[i] = i;
    }

    ops::add_cpu(*a, *b, *result);

    for (int i = 0; i < 4; ++i) {
        assert(result->data()[i] == 2 * i);
    }

    std::cout << "CPU addition test passed." << std::endl;
}

int main() {
    test_add_cpu();
    return 0;
}