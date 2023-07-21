#include "MathUtil.h"
#include "LogUtil.h"
#include "utils/DefUtil.h"
#include <iostream>
#include <time.h>
// const enum eRotationOrder gRotationOrder = eRotationOrder::XYZ;
// const tVector gGravity = tVector4(0, -9.8, 0, 0);
// const tVector gGravity = tVector4(0, 0, 0, 0);
cRand cMathUtil::gRand = cRand();

bool cMathUtil::IsPoint(const tVector4 &vec)
{
    return std::fabs(vec[3] - 1.0) < 1e-10;
}
tVector4 cMathUtil::VecToPoint(const tVector4 &vec)
{
    tVector4 new_vec = vec;
    new_vec[3] = 1;
    return new_vec;
}
int cMathUtil::Clamp(int val, int min, int max)
{
    return SIM_MAX(min, SIM_MIN(val, max));
}

// void cMathUtil::Clamp(const tVectorXd &min, const tVectorXd &max,
//                       tVectorXd &out_vec)
// {
//     out_vec = out_vec.cwiseMin(max).cwiseMax(min);
// }

_FLOAT cMathUtil::Clamp(_FLOAT val, _FLOAT min, _FLOAT max)
{
    return SIM_MAX(min, SIM_MIN(val, max));
}

_FLOAT cMathUtil::Saturate(_FLOAT val) { return Clamp(val, 0.0, 1.0); }

_FLOAT cMathUtil::Lerp(_FLOAT t, _FLOAT val0, _FLOAT val1)
{
    return (1 - t) * val0 + t * val1;
}

_FLOAT cMathUtil::NormalizeAngle(_FLOAT theta)
{
    // normalizes theta to be between [-pi, pi]
    _FLOAT norm_theta = fmod(theta, 2 * M_PI);
    if (norm_theta > M_PI)
    {
        norm_theta = -2 * M_PI + norm_theta;
    }
    else if (norm_theta < -M_PI)
    {
        norm_theta = 2 * M_PI + norm_theta;
    }
    return norm_theta;
}

_FLOAT cMathUtil::RandFloat() { return RandFloat(0, 1); }

_FLOAT cMathUtil::RandFloat(_FLOAT min, _FLOAT max)
{
    return gRand.RandFloat(min, max);
}

_FLOAT cMathUtil::RandFloatNorm(_FLOAT mean, _FLOAT stdev)
{
    return gRand.RandFloatNorm(mean, stdev);
}

_FLOAT cMathUtil::RandFloatExp(_FLOAT lambda)
{
    return gRand.RandFloatExp(lambda);
}

_FLOAT cMathUtil::RandFloatSeed(_FLOAT seed)
{
    unsigned int int_seed = *reinterpret_cast<unsigned int *>(&seed);
    std::default_random_engine rand_gen(int_seed);
    std::uniform_real_distribution<_FLOAT> dist;
    return dist(rand_gen);
}

int cMathUtil::RandInt() { return gRand.RandInt(); }

int cMathUtil::RandInt(int min, int max) { return gRand.RandInt(min, max); }

int cMathUtil::RandUint() { return gRand.RandUint(); }

int cMathUtil::RandUint(unsigned int min, unsigned int max)
{
    return gRand.RandUint(min, max);
}

int cMathUtil::RandIntExclude(int min, int max, int exc)
{
    return gRand.RandIntExclude(min, max, exc);
}

void cMathUtil::SeedRand(unsigned long int seed)
{
    gRand.Seed(seed);
    srand(gRand.RandInt());
}

int cMathUtil::RandSign() { return gRand.RandSign(); }

_FLOAT cMathUtil::SmoothStep(_FLOAT t)
{
    _FLOAT val = t * t * t * (t * (t * 6 - 15) + 10);
    return val;
}

bool cMathUtil::FlipCoin(_FLOAT p) { return gRand.FlipCoin(p); }

_FLOAT cMathUtil::Sign(_FLOAT val) { return SignAux<_FLOAT>(val); }

int cMathUtil::Sign(int val) { return SignAux<int>(val); }

_FLOAT cMathUtil::AddAverage(_FLOAT avg0, int count0, _FLOAT avg1, int count1)
{
    _FLOAT total = count0 + count1;
    return (count0 / total) * avg0 + (count1 / total) * avg1;
}

tVector4 cMathUtil::AddAverage(const tVector4 &avg0, int count0,
                               const tVector4 &avg1, int count1)
{
    _FLOAT total = count0 + count1;
    return (count0 / total) * avg0 + (count1 / total) * avg1;
}

// void cMathUtil::AddAverage(const tVectorXd &avg0, int count0,
//                            const tVectorXd &avg1, int count1,
//                            tVectorXd &out_result)
// {
//     FLOAT total = count0 + count1;
//     out_result = (count0 / total) * avg0 + (count1 / total) * avg1;
// }

// void cMathUtil::CalcSoftmax(const tVectorXd &vals, FLOAT temp,
//                             tVectorXd &out_prob)
// {
//     assert(out_prob.size() == vals.size());
//     int num_vals = static_cast<int>(vals.size());
//     FLOAT sum = 0;
//     FLOAT max_val = vals.maxCoeff();
//     for (int i = 0; i < num_vals; ++i)
//     {
//         FLOAT val = vals[i];
//         val = std::exp((val - max_val) / temp);
//         out_prob[i] = val;
//         sum += val;
//     }

//     out_prob /= sum;
// }

// FLOAT cMathUtil::EvalGaussian(const tVectorXd &mean,
//                                const tVectorXd &covar,
//                                const tVectorXd &sample)
// {
//     assert(mean.size() == covar.size());
//     assert(sample.size() == covar.size());

//     tVectorXd diff = sample - mean;
//     FLOAT exp_val = diff.dot(diff.cwiseQuotient(covar));
//     FLOAT likelihood = std::exp(-0.5 * exp_val);

//     FLOAT partition = CalcGaussianPartition(covar);
//     likelihood /= partition;
//     return likelihood;
// }

// FLOAT cMathUtil::EvalGaussian(FLOAT mean, FLOAT covar, FLOAT sample)
// {
//     FLOAT diff = sample - mean;
//     FLOAT exp_val = diff * diff / covar;
//     FLOAT norm = 1 / std::sqrt(2 * M_PI * covar);
//     FLOAT likelihood = norm * std::exp(-0.5 * exp_val);
//     return likelihood;
// }

// FLOAT cMathUtil::CalcGaussianPartition(const tVectorXd &covar)
// {
//     int data_size = static_cast<int>(covar.size());
//     FLOAT det = covar.prod();
//     FLOAT partition = std::sqrt(std::pow(2 * M_PI, data_size) * det);
//     return partition;
// }

// FLOAT cMathUtil::EvalGaussianLogp(const tVectorXd &mean,
//                                    const tVectorXd &covar,
//                                    const tVectorXd &sample)
// {
//     int data_size = static_cast<int>(covar.size());

//     tVectorXd diff = sample - mean;
//     FLOAT logp = -0.5 * diff.dot(diff.cwiseQuotient(covar));
//     FLOAT det = covar.prod();
//     logp += -0.5 * (data_size * std::log(2 * M_PI) + std::log(det));

//     return logp;
// }

// FLOAT cMathUtil::EvalGaussianLogp(FLOAT mean, FLOAT covar, FLOAT sample)
// {
//     FLOAT diff = sample - mean;
//     FLOAT logp = -0.5 * diff * diff / covar;
//     logp += -0.5 * (std::log(2 * M_PI) + std::log(covar));
//     return logp;
// }

// FLOAT cMathUtil::Sigmoid(FLOAT x) { return Sigmoid(x, 1, 0); }

// FLOAT cMathUtil::Sigmoid(FLOAT x, FLOAT gamma, FLOAT bias)
// {
//     FLOAT exp = -gamma * (x + bias);
//     FLOAT val = 1 / (1 + std::exp(exp));
//     return val;
// }

// int cMathUtil::SampleDiscreteProb(const tVectorX &probs)
// {
//     assert(std::abs(probs.sum() - 1) < 0.00001);
//     FLOAT rand = RandFloat();

//     int rand_idx = gInvalidIdx;
//     int num_probs = static_cast<int>(probs.size());
//     for (int i = 0; i < num_probs; ++i)
//     {
//         FLOAT curr_prob = probs[i];
//         rand -= curr_prob;

//         if (rand <= 0)
//         {
//             rand_idx = i;
//             break;
//         }
//     }
//     return rand_idx;
// }

/**
 * \briewf          categorical random distribution
 */
int cMathUtil::RandIntCategorical(const std::vector<_FLOAT> &prop)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(prop.begin(), prop.end());
    int num = d(gen);
    return num;
}

tVector4 cMathUtil::CalcBarycentric(const tVector4 &p, const tVector4 &a,
                                    const tVector4 &b, const tVector4 &c)
{
    tVector4 v0 = b - a;
    tVector4 v1 = c - a;
    tVector4 v2 = p - a;

    _FLOAT d00 = v0.dot(v0);
    _FLOAT d01 = v0.dot(v1);
    _FLOAT d11 = v1.dot(v1);
    _FLOAT d20 = v2.dot(v0);
    _FLOAT d21 = v2.dot(v1);
    _FLOAT denom = d00 * d11 - d01 * d01;
    _FLOAT v = (d11 * d20 - d01 * d21) / denom;
    _FLOAT w = (d00 * d21 - d01 * d20) / denom;
    _FLOAT u = 1.0f - v - w;
    return tVector4(u, v, w, 0);
}

bool cMathUtil::ContainsAABB(const tVector4 &pt, const tVector4 &aabb_min,
                             const tVector4 &aabb_max)
{
    bool contains = pt[0] >= aabb_min[0] && pt[1] >= aabb_min[1] &&
                    pt[2] >= aabb_min[2] && pt[0] <= aabb_max[0] &&
                    pt[1] <= aabb_max[1] && pt[2] <= aabb_max[2];
    return contains;
}

bool cMathUtil::ContainsAABB(const tVector4 &aabb_min0,
                             const tVector4 &aabb_max0,
                             const tVector4 &aabb_min1,
                             const tVector4 &aabb_max1)
{
    return ContainsAABB(aabb_min0, aabb_min1, aabb_max1) &&
           ContainsAABB(aabb_max0, aabb_min1, aabb_max1);
}

bool cMathUtil::ContainsAABBXZ(const tVector4 &pt, const tVector4 &aabb_min,
                               const tVector4 &aabb_max)
{
    bool contains = pt[0] >= aabb_min[0] && pt[2] >= aabb_min[2] &&
                    pt[0] <= aabb_max[0] && pt[2] <= aabb_max[2];
    return contains;
}

bool cMathUtil::ContainsAABBXZ(const tVector4 &aabb_min0,
                               const tVector4 &aabb_max0,
                               const tVector4 &aabb_min1,
                               const tVector4 &aabb_max1)
{
    return ContainsAABBXZ(aabb_min0, aabb_min1, aabb_max1) &&
           ContainsAABBXZ(aabb_max0, aabb_min1, aabb_max1);
}

void cMathUtil::CalcAABBIntersection(const tVector4 &aabb_min0,
                                     const tVector4 &aabb_max0,
                                     const tVector4 &aabb_min1,
                                     const tVector4 &aabb_max1,
                                     tVector4 &out_min, tVector4 &out_max)
{
    out_min = aabb_min0.cwiseMax(aabb_min1);
    out_max = aabb_max0.cwiseMin(aabb_max1);
    if (out_min[0] > out_max[0])
    {
        out_min[0] = 0;
        out_max[0] = 0;
    }
    if (out_min[1] > out_max[1])
    {
        out_min[1] = 0;
        out_max[1] = 0;
    }
    if (out_min[2] > out_max[2])
    {
        out_min[2] = 0;
        out_max[2] = 0;
    }
}

void cMathUtil::CalcAABBUnion(const tVector4 &aabb_min0,
                              const tVector4 &aabb_max0,
                              const tVector4 &aabb_min1,
                              const tVector4 &aabb_max1, tVector4 &out_min,
                              tVector4 &out_max)
{
    out_min = aabb_min0.cwiseMin(aabb_min1);
    out_max = aabb_max0.cwiseMax(aabb_max1);
}

bool cMathUtil::IntersectAABB(const tVector4 &aabb_min0,
                              const tVector4 &aabb_max0,
                              const tVector4 &aabb_min1,
                              const tVector4 &aabb_max1)
{
    tVector4 center0 = 0.5 * (aabb_max0 + aabb_min0);
    tVector4 center1 = 0.5 * (aabb_max1 + aabb_min1);
    tVector4 size0 = aabb_max0 - aabb_min0;
    tVector4 size1 = aabb_max1 - aabb_min1;
    tVector4 test_len = 0.5 * (size0 + size1);
    tVector4 delta = center1 - center0;
    bool overlap = (std::abs(delta[0]) <= test_len[0]) &&
                   (std::abs(delta[1]) <= test_len[1]) &&
                   (std::abs(delta[2]) <= test_len[2]);
    return overlap;
}

bool cMathUtil::IntersectAABBXZ(const tVector4 &aabb_min0,
                                const tVector4 &aabb_max0,
                                const tVector4 &aabb_min1,
                                const tVector4 &aabb_max1)
{
    tVector4 center0 = 0.5 * (aabb_max0 + aabb_min0);
    tVector4 center1 = 0.5 * (aabb_max1 + aabb_min1);
    tVector4 size0 = aabb_max0 - aabb_min0;
    tVector4 size1 = aabb_max1 - aabb_min1;
    tVector4 test_len = 0.5 * (size0 + size1);
    tVector4 delta = center1 - center0;
    bool overlap = (std::abs(delta[0]) <= test_len[0]) &&
                   (std::abs(delta[2]) <= test_len[2]);
    return overlap;
}

bool cMathUtil::CheckNextInterval(_FLOAT delta, _FLOAT curr_val,
                                  _FLOAT int_size)
{
    _FLOAT pad = 0.001 * delta;
    int curr_count = static_cast<int>(std::floor((curr_val + pad) / int_size));
    int prev_count =
        static_cast<int>(std::floor((curr_val + pad - delta) / int_size));
    bool new_action = (curr_count != prev_count);
    return new_action;
}

tVector4 cMathUtil::SampleRandPt(const tVector4 &bound_min,
                                 const tVector4 &bound_max)
{
    tVector4 pt = tVector4(RandFloat(bound_min[0], bound_max[0]),
                           RandFloat(bound_min[1], bound_max[1]),
                           RandFloat(bound_min[2], bound_max[2]), 0);
    return pt;
}

tVector4 cMathUtil::SampleRandPtBias(const tVector4 &bound_min,
                                     const tVector4 &bound_max)
{
    return SampleRandPtBias(bound_min, bound_max,
                            0.5 * (bound_max + bound_min));
}

tVector4 cMathUtil::SampleRandPtBias(const tVector4 &bound_min,
                                     const tVector4 &bound_max,
                                     const tVector4 &focus)
{
    _FLOAT t = RandFloat(0, 1);
    tVector4 size = bound_max - bound_min;
    tVector4 new_min = focus + (t * 0.5) * size;
    tVector4 new_max = focus - (t * 0.5) * size;
    tVector4 offset = (bound_min - new_min).cwiseMax(0);
    offset += (bound_max - new_max).cwiseMin(0);
    new_min += offset;
    new_max += offset;

    return SampleRandPt(new_min, new_max);
}

// tQuaterniondd cMathUtil::RotMatToQuaternion(const tMatrix &mat)
//{
//	//
// http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
//	FLOAT eta = 0;
//	FLOAT q1, q2, q3, q4;	// = [w, x, y, z]
//
//	// determine q1
//	{
//		FLOAT detect_value = mat(0, 0) + mat(1, 1) + mat(2, 2);
//		if (detect_value > eta)
//		{
//			q1 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			FLOAT numerator = 0;
//			numerator += std::pow(mat(2, 1) - mat(1, 2), 2);
//			numerator += std::pow(mat(0, 2) - mat(2, 0), 2);
//			numerator += std::pow(mat(1, 0) - mat(0, 1), 2);
//			q1 = 0.5 *  std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	// determine q2
//	{
//		FLOAT detect_value = mat(0, 0) - mat(1, 1) - mat(2, 2);
//		if (detect_value > eta)
//		{
//			q2 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			FLOAT numerator = 0;
//			numerator += std::pow(mat(2, 1) - mat(1, 2), 2);
//			numerator += std::pow(mat(0, 1) + mat(1, 0), 2);
//			numerator += std::pow(mat(2, 0) + mat(0, 2), 2);
//			q2 = 0.5 * std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	// determine q3
//	{
//		FLOAT detect_value = -mat(0, 0) + mat(1, 1) - mat(2, 2);
//		if (detect_value > eta)
//		{
//			q3 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			FLOAT numerator = 0;
//			numerator += std::pow(mat(0, 2) - mat(2, 0), 2);
//			numerator += std::pow(mat(0, 1) + mat(1, 0), 2);
//			numerator += std::pow(mat(1, 2) + mat(2, 1), 2);
//			q3 = 0.5 * std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	// determine q4
//	{
//		FLOAT detect_value = -mat(0, 0) - mat(1, 1) + mat(2, 2);
//		if (detect_value > eta)
//		{
//			q4 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			FLOAT numerator = 0;
//			numerator += std::pow(mat(1, 0) - mat(0, 1), 2);
//			numerator += std::pow(mat(2, 0) + mat(0, 2), 2);
//			numerator += std::pow(mat(2, 1) + mat(1, 2), 2);
//			q4 = 0.5 * std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	return tQuaterniondd(q1, q2, q3, q4);
//}

tEigenArr<tVector3> cMathUtil::ExpandTriangle(const tVector3 &v0,
                                              const tVector3 &v1,
                                              const tVector3 &v2, _FLOAT h)
{
    auto calc_height_vec_from_edge_to_pt =
        [](const tVector3 &pt, const tVector3 &ev1, const tVector3 &ev2)
    {
        tVector3 hvec = pt - ev1 +
                        (ev2 - ev1) * (ev2 - ev1).dot(ev1 - pt) /
                            (ev2 - ev1).dot(ev2 - ev1);
        hvec.normalize();
        return hvec;
    };
    tEigenArr<tVector3> raw_pt = {v0, v1, v2};
    tEigenArr<tVector3> new_edges_vt_lst = {};
    tEigenArr<tVector3> new_pt = {};

    for (int i = 0; i < 3; i++)
    {
        tVector3 p = raw_pt[i];
        tVector3 ev0 = raw_pt[(i + 1) % 3];
        tVector3 ev1 = raw_pt[(i + 2) % 3];
        tVector3 height_vec = calc_height_vec_from_edge_to_pt(p, ev0, ev1);
        // std::cout << "hvec = " << height_vec.transpose() << std::endl;
        new_edges_vt_lst.push_back(ev0 - h * height_vec);
        new_edges_vt_lst.push_back(ev1 - h * height_vec);
    }

    for (int i = 2; i < 5; i++)
    {
        tVector3 a = new_edges_vt_lst[2 * (i % 3) + 0],
                 b = new_edges_vt_lst[2 * (i % 3) + 1];
        tVector3 c = new_edges_vt_lst[2 * ((i + 1) % 3) + 0],
                 d = new_edges_vt_lst[2 * ((i + 1) % 3) + 1];
        // std::cout << "a = " << a.transpose() << std::endl;
        // std::cout << "b = " << b.transpose() << std::endl;
        // std::cout << "c = " << c.transpose() << std::endl;
        // std::cout << "d = " << d.transpose() << std::endl;
        tVector3 inter_pt = cMathUtil::CalcTwoLineIntersection(a, b, c, d);
        // std::cout << "pt = " << inter_pt.transpose() << std::endl;
        new_pt.push_back(inter_pt);
    }
    return new_pt;
}

tVector3 cMathUtil::CalcTwoLineIntersection(const tVector3 &a,
                                            const tVector3 &b,
                                            const tVector3 &c,
                                            const tVector3 &d)
{
    // 1. if some points are the same
    bool is_same_ac = cMathUtil::IsSame(a, c, 1e-6);
    bool is_same_ad = cMathUtil::IsSame(a, d, 1e-6);
    bool is_same_bc = cMathUtil::IsSame(b, c, 1e-6);
    bool is_same_bd = cMathUtil::IsSame(b, d, 1e-6);
    if (is_same_ac || is_same_ad)
    {
        return a;
    }
    else if (is_same_bc || is_same_bc)
    {
        return b;
    }
    else
    {
        // 1. assert the same plane
        tVector3 n0 = ((b - a).cross(c - b)).normalized();
        tVector3 n1 = ((d - c).cross(c - a)).normalized();

        // assert normal
        if ((n0 - n1).cwiseAbs().maxCoeff() > 1e-2)
        {
            SIM_ASSERT((n0 + n1).cwiseAbs().maxCoeff() < 1e-2);
        }

        // 2. calc
        Eigen::Matrix<_FLOAT, 3, 2> A;
        A.col(0) = b - a;
        A.col(1) = c - d;
        tVector3 B = c - a;
        _FLOAT t1 = ((A.transpose() * A).inverse() * (A.transpose() * B))[0];
        return a + (b - a) * t1;
    }
}