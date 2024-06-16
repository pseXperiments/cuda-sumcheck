#include <cassert>
#include <iostream>
#include <cstring>

#include "prime_field.h"
#include "./barretenberg/ecc/curves/bn254/fq.hpp"

int main() {
    bb::fq a = bb::fq(1UL);
    bb::fq b = bb::fq(2U);
    bb::fq c = a + b;
    assert(c == bb::fq(3U));
    c = a * b;
    assert(c == bb::fq(2U));
    // memory layout test
    struct Field val_c { { 1UL, 2UL, 3UL, 4UL } };
    bb::fq val;
    std::memcpy(&val, &val_c, 32);
    assert(val.data[0] == 1UL);
    assert(val.data[1] == 2UL);
    assert(val.data[2] == 3UL);
    assert(val.data[3] == 4UL);
    std::cout << "test ended" << std::endl;
    return 0;
}
