#include "neural_core/rng.hpp"

#include <atomic>
#include <limits>
#include <random>
#include <stdexcept>

using std::atomic;
using std::memory_order_acquire;
using std::memory_order_relaxed;
using std::memory_order_release;
using std::mt19937;
using std::normal_distribution;
using std::numeric_limits;
using std::random_device;
using std::runtime_error;
using std::seed_seq;
using std::size_t;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

// deterministic per-thread streams with global seed + per-thread stream id
static atomic<uint32_t> g_seed_value{0};
static atomic<bool> g_use_deterministic_seed{false};
static atomic<uint32_t> g_seed_epoch{0};

thread_local uint32_t t_stream_id = 0;
thread_local bool t_stream_id_set = false;

void set_thread_stream_id(uint32_t stream_id)
{
    t_stream_id = stream_id;
    t_stream_id_set = true;
}

void set_global_seed(uint32_t seed)
{
    g_seed_value.store(seed, memory_order_relaxed);
    g_use_deterministic_seed.store(true, memory_order_relaxed);
    g_seed_epoch.fetch_add(1, memory_order_release);
}

void set_nondeterministic_seed()
{
    g_use_deterministic_seed.store(false, memory_order_relaxed);
    g_seed_epoch.fetch_add(1, memory_order_release);
}

static mt19937& thread_rng()
{
    thread_local mt19937 rng;
    thread_local uint32_t local_epoch = numeric_limits<uint32_t>::max();

    const uint32_t current_epoch = g_seed_epoch.load(memory_order_acquire);
    if (local_epoch != current_epoch) {
        local_epoch = current_epoch;

        if (g_use_deterministic_seed.load(memory_order_relaxed)) {
            if (!t_stream_id_set) {
                throw runtime_error(
                    "thread_rng: deterministic mode requires set_thread_stream_id() to be called once per thread before any random draws");
            }

            const uint32_t seed = g_seed_value.load(memory_order_relaxed);
            seed_seq seq{seed, current_epoch, t_stream_id, 0x9e3779b9u, 0x85ebca6bu};
            rng.seed(seq);
        } else {
            const uint32_t stream_id = t_stream_id_set ? t_stream_id : 0u;

            random_device rd;
            seed_seq seq{rd(), rd(), stream_id, 0x9e3779b9u, 0x85ebca6bu};
            rng.seed(seq);
        }
    }

    return rng;
}

double random_gaussian()
{
    thread_local normal_distribution<double> dist(0.0, 1.0);

    thread_local uint32_t dist_epoch = numeric_limits<uint32_t>::max();
    const uint32_t current_epoch = g_seed_epoch.load(memory_order_acquire);
    if (dist_epoch != current_epoch) {
        dist_epoch = current_epoch;
        dist.reset();
    }

    return dist(thread_rng());
}

double random_uniform()
{
    thread_local uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(thread_rng());
}

size_t random_uniform_int(size_t min_value, size_t max_value)
{
    if (min_value > max_value) {
        throw runtime_error("random_uniform_int: min_value cannot exceed max_value");
    }
    uniform_int_distribution<size_t> dist(min_value, max_value);
    return dist(thread_rng());
}
