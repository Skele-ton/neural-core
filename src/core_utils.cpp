#include "neural_core/core_utils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using std::abs;
using std::numeric_limits;
using std::round;
using std::runtime_error;
using std::size_t;

bool is_whole_number(double v, double epsilon)
{
    return abs(v - round(v)) <= epsilon;
}

void multiplication_overflow_check(const size_t a, const size_t b, const char* error_msg)
{
    if (a != 0 && b > numeric_limits<size_t>::max() / a) {
        throw runtime_error(error_msg);
    }
}
