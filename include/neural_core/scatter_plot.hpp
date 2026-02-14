#pragma once

#include <string>

#include "neural_core/matrix.hpp"

void scatter_plot(const std::string& path, const Matrix& points, const Matrix& labels = Matrix());
