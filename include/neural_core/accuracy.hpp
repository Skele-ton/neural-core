#pragma once

#include <cstddef>

#include "neural_core/matrix.hpp"

class Accuracy
{
public:
    virtual ~Accuracy() = default;
    virtual void init(const Matrix& y_true);
    virtual void reset();

    double calculate(const Matrix& y_pred, const Matrix& y_true);
    double calculate_accumulated() const;

    void new_pass();

protected:
    virtual std::size_t compare(const Matrix& y_pred, const Matrix& y_true) = 0;

private:
    std::size_t accumulated_sum = 0;
    std::size_t accumulated_count = 0;
};

class AccuracyCategorical : public Accuracy
{
public:
    explicit AccuracyCategorical(bool binary = false);

    bool get_binray() const;

protected:
    std::size_t compare(const Matrix& y_pred, const Matrix& y_true) override;

private:
    bool binary;
};

class AccuracyRegression : public Accuracy
{
public:
    explicit AccuracyRegression(double precision_divisor = 250.0);

    void init(const Matrix& y_true) override;
    void reset() override;

    double get_precision_divisor() const;

protected:
    std::size_t compare(const Matrix& y_pred, const Matrix& y_true) override;

private:
    double precision_divisor;
    double precision;
    bool initialized;
};
