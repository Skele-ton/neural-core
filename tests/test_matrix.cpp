#include "tests/test_common.hpp"

TEST_CASE("Matrix construction, assignment, and access")
{
    Matrix m(2, 3, 1.5);
    CHECK(m.get_rows() == 2);
    CHECK(m.get_cols() == 3);
    CHECK(m.get_data().size() == m.get_rows() * m.get_cols());
    CHECK(m(0, 0) == doctest::Approx(1.5));
    CHECK(m(1, 2) == doctest::Approx(1.5));

    m(0, 1) = 4.2;
    CHECK(m(0, 1) == doctest::Approx(4.2));

    m.assign(3, 2, 0.0);
    CHECK(m.get_rows() == 3);
    CHECK(m.get_cols() == 2);
    CHECK(m(2, 1) == doctest::Approx(0.0));
}

TEST_CASE("Matrix operator() throws on out-of-bounds access")
{
    Matrix m(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(m(1, 0),
                         "Matrix::operator(): index out of bounds",
                         runtime_error);

    const Matrix mc(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(mc(0, 1),
                         "Matrix::operator() const: index out of bounds",
                         runtime_error);
}

TEST_CASE("Matrix check helpers detect empty matrices and row/column vectors")
{
    Matrix empty;
    CHECK(empty.is_empty());

    Matrix row(1, 3, 0.0);
    CHECK_FALSE(row.is_empty());
    CHECK(row.is_row_vector());
    CHECK_FALSE(row.is_col_vector());
    CHECK(row.is_vector());

    Matrix col(3, 1, 0.0);
    CHECK(col.is_col_vector());
    CHECK_FALSE(col.is_row_vector());
    CHECK(col.is_vector());

    Matrix box(2, 2, 0.0);
    CHECK_FALSE(box.is_vector());
}

TEST_CASE("Matrix require_* helpers validate shape and emptiness")
{
    Matrix empty;
    CHECK_THROWS_WITH_AS(empty.require_non_empty("empty"),
                         "empty",
                         runtime_error);

    Matrix m(2, 3, 0.0);
    CHECK_THROWS_WITH_AS(m.require_rows(1, "rows"),
                         "rows",
                         runtime_error);
    CHECK_NOTHROW(m.require_rows(2, "rows"));
    CHECK_THROWS_WITH_AS(m.require_cols(4, "cols"),
                         "cols",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.require_shape(2, 2, "shape"),
                         "shape",
                         runtime_error);
}

TEST_CASE("Matrix print writes expected output")
{
    ostringstream oss;
    auto* old_buf = cout.rdbuf(oss.rdbuf());

    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.5;
    m.print();

    cout.rdbuf(old_buf);
    CHECK(oss.str() == "1 2\n3 4.5\n");
}

TEST_CASE("Matrix scale_by_scalar scales values and rejects a value of zero")
{
    Matrix m(1, 2);
    m(0, 0) = 2.0;
    m(0, 1) = 4.0;
    m.scale_by_scalar(2);
    CHECK(m(0, 0) == doctest::Approx(1.0));
    CHECK(m(0, 1) == doctest::Approx(2.0));

    Matrix bad(1, 1, 2.0);
    CHECK_THROWS_WITH_AS(bad.scale_by_scalar(0),
                         "Matrix::scale_by_scalar: value cannot be zero 0",
                         runtime_error);
}

TEST_CASE("Matrix transpose and argmax handle empty and non-empty cases")
{
    Matrix empty;
    CHECK(empty.transpose().is_empty());
    CHECK(empty.argmax().is_empty());

    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 0.5;

    Matrix t = m.transpose();
    CHECK(t.get_rows() == 3);
    CHECK(t.get_cols() == 2);
    CHECK(t(0, 0) == doctest::Approx(1.0));
    CHECK(t(2, 1) == doctest::Approx(0.5));

    Matrix arg = m.argmax();
    CHECK(arg.get_rows() == 2);
    CHECK(arg.get_cols() == 1);
    CHECK(arg(0, 0) == doctest::Approx(2.0));
    CHECK(arg(1, 0) == doctest::Approx(1.0));
}

TEST_CASE("Matrix slice_rows and slice_cols work and validate bounds")
{
    Matrix m(3, 4);
    double v = 1.0;
    for (size_t i = 0; i < m.get_rows(); ++i) {
        for (size_t j = 0; j < m.get_cols(); ++j) {
            m(i, j) = v++;
        }
    }

    Matrix rows = m.slice_rows(1, 3);
    CHECK(rows.get_rows() == 2);
    CHECK(rows.get_cols() == 4);
    CHECK(rows(0, 0) == doctest::Approx(m(1, 0)));
    CHECK(rows(1, 3) == doctest::Approx(m(2, 3)));

    Matrix cols = m.slice_cols(1, 3);
    CHECK(cols.get_rows() == 3);
    CHECK(cols.get_cols() == 2);
    CHECK(cols(0, 0) == doctest::Approx(m(0, 1)));
    CHECK(cols(2, 1) == doctest::Approx(m(2, 2)));

    CHECK_THROWS_WITH_AS(m.slice_rows(2, 1),
                         "Matrix::slice_rows: invalid slice bounds",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.slice_rows(0, 4),
                         "Matrix::slice_rows: invalid slice bounds",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.slice_cols(2, 1),
                         "Matrix::slice_cols: invalid slice bounds",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.slice_cols(0, 5),
                         "Matrix::slice_cols: invalid slice bounds",
                         runtime_error);
}

TEST_CASE("Matrix as_size_t validates integer-like values")
{
    Matrix m(1, 2);
    m(0, 0) = 2.0;
    m(0, 1) = 2.5;
    CHECK(m.as_size_t(0, 0) == static_cast<size_t>(2));

    CHECK_THROWS_WITH_AS(m.as_size_t(0, 1),
                         "Matrix::as_size_t: value is not integer-like",
                         runtime_error);

    Matrix n(1, 1);
    n(0, 0) = -1.0;
    CHECK_THROWS_WITH_AS(n.as_size_t(0, 0),
                         "Matrix::as_size_t: integer out of range",
                         runtime_error);
}

TEST_CASE("Matrix scalar_mean computes mean and rejects empty matrix")
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 3.0;
    m(1, 0) = 5.0; m(1, 1) = 7.0;
    CHECK(m.scalar_mean() == doctest::Approx(4.0));

    Matrix empty;
    CHECK_THROWS_WITH_AS(empty.scalar_mean(),
                         "Matrix::scalar_mean: cannot find mean of empty matrix",
                         runtime_error);
}

TEST_CASE("Matrix shuffle_rows_with validates inputs and shuffles with row/col labels")
{
    Matrix empty;
    Matrix y_empty(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(empty.shuffle_rows_with(y_empty),
                         "shuffle_rows_with: base matrix must be non-empty",
                         runtime_error);

    Matrix X(3, 2);
    X(0, 0) = 10.0; X(0, 1) = 11.0;
    X(1, 0) = 20.0; X(1, 1) = 21.0;
    X(2, 0) = 30.0; X(2, 1) = 31.0;

    Matrix bad_y(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(X.shuffle_rows_with(bad_y),
                         "shuffle_rows_with: y must be shape (1,N) or (N,1), where N = base matrix rows",
                         runtime_error);

    Matrix X_row = X;
    Matrix y_row(1, 3);
    y_row(0, 0) = 0.0; y_row(0, 1) = 1.0; y_row(0, 2) = 2.0;
    reset_deterministic_rng(1); // using a seed that ensures y_row actually gets shuffled
    X_row.shuffle_rows_with(y_row);
    for (size_t i = 0; i < X_row.get_rows(); ++i) {
        const size_t label = y_row.as_size_t(0, i);
        CHECK(X_row(i, 0) == doctest::Approx((label + 1) * 10.0));
        CHECK(X_row(i, 1) == doctest::Approx((label + 1) * 10.0 + 1.0));
    }

    Matrix X_col = X;
    Matrix y_col(3, 1);
    y_col(0, 0) = 0.0; y_col(1, 0) = 1.0; y_col(2, 0) = 2.0;
    reset_deterministic_rng(0);
    X_col.shuffle_rows_with(y_col);
    for (size_t i = 0; i < X_col.get_rows(); ++i) {
        const size_t label = y_col.as_size_t(i, 0);
        CHECK(X_col(i, 0) == doctest::Approx((label + 1) * 10.0));
        CHECK(X_col(i, 1) == doctest::Approx((label + 1) * 10.0 + 1.0));
    }

    // rows < 2 early return branch
    Matrix X_one(1, 2);
    X_one(0, 0) = 7.0; X_one(0, 1) = 8.0;
    Matrix y_one(1, 1, 3.0);
    X_one.shuffle_rows_with(y_one);
    CHECK(X_one(0, 0) == doctest::Approx(7.0));
    CHECK(y_one(0, 0) == doctest::Approx(3.0));
}

TEST_CASE("Matrix dot multiplies matrices and validates shapes")
{
    Matrix a(2, 3);
    a(0, 0) = 1.0; a(0, 1) = 2.0; a(0, 2) = 3.0;
    a(1, 0) = 4.0; a(1, 1) = 5.0; a(1, 2) = 6.0;

    Matrix b(3, 2);
    b(0, 0) = 7.0;  b(0, 1) = 8.0;
    b(1, 0) = 9.0;  b(1, 1) = 10.0;
    b(2, 0) = 11.0; b(2, 1) = 12.0;

    Matrix c = Matrix::dot(a, b);
    CHECK(c.get_rows() == 2);
    CHECK(c.get_cols() == 2);
    CHECK(c(0, 0) == doctest::Approx(58.0));
    CHECK(c(0, 1) == doctest::Approx(64.0));
    CHECK(c(1, 0) == doctest::Approx(139.0));
    CHECK(c(1, 1) == doctest::Approx(154.0));

    Matrix empty;
    CHECK_THROWS_WITH_AS(Matrix::dot(empty, b),
                         "Matrix::dot: matrices must not be empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(Matrix::dot(a, empty),
                         "Matrix::dot: matrices must not be empty",
                         runtime_error);

    Matrix bad(4, 1, 0.0);
    CHECK_THROWS_WITH_AS(Matrix::dot(a, bad),
                         "Matrix::dot: matrices have incompatible shapes",
                         runtime_error);
}

TEST_CASE("Matrix max_absolute_difference computes max difference and validates shape")
{
    Matrix a(2, 3);
    a(0, 0) = 1.0;  a(0, 1) = -2.0; a(0, 2) = 3.0;
    a(1, 0) = 4.0;  a(1, 1) = 5.0;  a(1, 2) = -6.0;

    Matrix b(2, 3);
    b(0, 0) = 1.25; b(0, 1) = -1.5; b(0, 2) = 2.0;
    b(1, 0) = 4.0;  b(1, 1) = 3.4;  b(1, 2) = -6.1;

    CHECK(Matrix::max_absolute_difference(a, b) == doctest::Approx(1.6));

    Matrix wrong_shape(3, 2, 0.0);
    CHECK_THROWS_WITH_AS(Matrix::max_absolute_difference(a, wrong_shape),
                         "Matrix::max_absolute_difference: shape mismatch",
                         runtime_error);
}

