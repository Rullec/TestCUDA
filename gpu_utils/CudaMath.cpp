#include "gpu_utils/CudaMath.h"

template <> bool cCudaMath::IsNan(const float &val) { return isnan(val); };
template <> bool cCudaMath::IsNan(const double &val) { return isnan(val); };