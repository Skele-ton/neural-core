#include "tests/test_common.hpp"

TEST_CASE("is_whole_number detects integer-like values")
{
    CHECK(is_whole_number(5.0));
    CHECK(is_whole_number(5.0 + 1e-8));
    CHECK_FALSE(is_whole_number(5.1));
}

TEST_CASE("multiplication_overflow_check throws on overflow and allows safe values")
{
    CHECK_NOTHROW(multiplication_overflow_check(100, 200, "overflow"));
    const size_t max = numeric_limits<size_t>::max();
    CHECK_THROWS_WITH_AS(multiplication_overflow_check(max, 2, "overflow"),
                         "overflow",
                         runtime_error);
}

TEST_CASE("random_gaussian produces finite values")
{
    reset_deterministic_rng(0);
    double v1 = random_gaussian();
    double v2 = random_gaussian();
    CHECK(isfinite(v1));
    CHECK(isfinite(v2));
}

TEST_CASE("random_uniform produces values in [0,1)")
{
    reset_deterministic_rng(42);
    double v1 = random_uniform();
    CHECK(v1 >= 0.0);
    CHECK(v1 < 1.0);
    double v2 = random_uniform();
    CHECK(v2 >= 0.0);
    CHECK(v2 < 1.0);
}

TEST_CASE("deterministic rng produces repeatable sequences across threads")
{
    set_global_seed(123);

    vector<double> seq_a(16);
    vector<double> seq_b(16);

    auto worker = [](uint32_t stream_id, vector<double>* out) {
        set_thread_stream_id(stream_id);
        for (size_t i = 0; i < out->size(); ++i) {
            (*out)[i] = random_uniform();
        }
    };

    thread t1(worker, 0u, &seq_a);
    thread t2(worker, 0u, &seq_b);
    t1.join();
    t2.join();

    for (size_t i = 0; i < seq_a.size(); ++i) {
        CHECK(seq_a[i] == doctest::Approx(seq_b[i]));
    }
}

TEST_CASE("random_uniform_int validates bounds")
{
    CHECK_THROWS_WITH_AS(random_uniform_int(5, 4),
                         "random_uniform_int: min_value cannot exceed max_value",
                         runtime_error);

    reset_deterministic_rng(7);
    size_t v1 = random_uniform_int(0, 10);
    CHECK(v1 <= 10);
}

TEST_CASE("thread_rng requires stream id in deterministic mode")
{
    set_global_seed(123);
    t_stream_id_set = false;

    CHECK_THROWS_WITH_AS(random_uniform(),
                         "thread_rng: deterministic mode requires set_thread_stream_id() to be called once per thread before any random draws",
                         runtime_error);

    set_thread_stream_id(0);

    CHECK_NOTHROW(random_uniform());
}

TEST_CASE("set_nondeterministic_seed disables deterministic requirement")
{
    // set_nondeterministic_seed without prior
    set_nondeterministic_seed();
    double v = random_uniform();
    CHECK(v >= 0.0);
    CHECK(v < 1.0);

    vector<double> seq_a(16);
    vector<double> seq_b(16);

    auto worker = [](uint32_t stream_id, vector<double>* out) {
        set_thread_stream_id(stream_id);
        for (size_t i = 0; i < out->size(); ++i) {
            (*out)[i] = random_uniform();
        }
    };

    thread t1(worker, 0u, &seq_a);
    thread t2(worker, 1u, &seq_b);
    t1.join();
    t2.join();

    bool any_diff = false;
    for (size_t i = 0; i < seq_a.size(); ++i) {
        if (seq_a[i] != seq_b[i]) {
            any_diff = true;
            break;
        }
    }

    CHECK(any_diff == true);

    // set_nondeterministic_seed to re-seed
    set_nondeterministic_seed();
    thread t3(worker, 0u, &seq_a);
    thread t4(worker, 1u, &seq_b);
    t3.join();
    t4.join();

    bool any_diff_after = false;
    for (size_t i = 0; i < seq_a.size(); ++i) {
        if (seq_a[i] != seq_b[i]) {
            any_diff_after = true;
            break;
        }
    }
    CHECK(any_diff_after == true);
}

