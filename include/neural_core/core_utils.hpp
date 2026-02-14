#pragma once

#include <cstddef>

bool is_whole_number(double v, double epsilon = 1e-7);
void multiplication_overflow_check(std::size_t a, std::size_t b, const char* error_msg);
