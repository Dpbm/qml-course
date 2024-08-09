#pragma once

#include <cuComplex.h>
#include <cmath>

const double half_superposition = 1.0/std::sqrt(2.0);

cuDoubleComplex H[] = {
    {half_superposition, 0.0}, {half_superposition, 0.0},
    {half_superposition, 0.0}, {-half_superposition, 0.0}
};
