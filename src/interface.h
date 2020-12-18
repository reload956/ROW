#ifdef __cplusplus

#include <cstdlib>

#else
#include <stdlib.h>
#endif

#include "config.h"

#pragma once

EXTERN_C double omp_accumulate_false_sharing(size_t count);
EXTERN_C double omp_accumulate_aligned(size_t count);
EXTERN_C double omp_accumulate_atomic(size_t count);
EXTERN_C double omp_accumulate_critical(size_t count);
EXTERN_C double omp_accumulate_reduction_static(size_t count);
EXTERN_C double omp_accumulate_reduction_dynamic(size_t count);

double cpp_accumulate_false_sharing(size_t count);

double cpp_accumulate_aligned(size_t count);

double cpp_accumulate_atomic(size_t count);

double cpp_accumulate_mutex(size_t count);

double cpp_accumulate_reduction_static(size_t count);
