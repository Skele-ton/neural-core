#pragma once

#include <cstddef>
#include <cstdint>

extern thread_local uint32_t t_stream_id;
extern thread_local bool t_stream_id_set;

void set_thread_stream_id(uint32_t stream_id);
void set_global_seed(uint32_t seed);
void set_nondeterministic_seed();

double random_gaussian();
double random_uniform();
std::size_t random_uniform_int(std::size_t min_value, std::size_t max_value);
