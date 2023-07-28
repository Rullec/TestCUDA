#include "FullConnectedNetSingleScalar.h"
#include "FullConnectedNetSingleScalarGPU.h"
#include "utils/DefUtil.h"
#include "utils/EigenUtil.h"
#include "utils/TensorUtil.h"

extern std::vector<cFCNetworkSingleScalarPtr> mNet1dLstCPU, mNet2dLstCPU;
extern std::vector<cFCNetworkSingleScalarGPUPtr> mNet1dLstGPU, mNet2dLstGPU;
#include "utils/TimeUtil.hpp"
void energy_verify()
{

    // energy debug
    if (true)
    {
        // test 1d
        int test_n = 5;
        auto net_cpu = mNet1dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<_FLOAT> E_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(1);
            batch_x.emplace_back(x);
            E_cpu_arr.push_back(net_cpu->forward_unnormed(x));
        }
        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet1dLstGPU[0];

        std::vector<_FLOAT> E_gpu = net_gpu->forward_unnormed_batch(batch_x);

        cTimeUtil::Begin("gpu");
        E_gpu = net_gpu->forward_unnormed_batch(batch_x);
        _FLOAT gpu_elasped = cTimeUtil::End("gpu");
        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < test_n; i++)
            printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
        printf("[1d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
    }
    else
    {
        // test 2d
        int test_n = 10000;
        auto net_cpu = mNet2dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<_FLOAT> E_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(2);
            batch_x.emplace_back(x);
            E_cpu_arr.push_back(net_cpu->forward_unnormed(x));
        }
        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet2dLstGPU[0];
        std::vector<_FLOAT> E_gpu = net_gpu->forward_unnormed_batch(batch_x);

        cTimeUtil::Begin("gpu");
        E_gpu = net_gpu->forward_unnormed_batch(batch_x);
        _FLOAT gpu_elasped = cTimeUtil::End("gpu");
        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < test_n; i++)
            printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
        printf("[2d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
    }
}

void energy_grad_verify()
{
    bool test_1d = false;
    if (test_1d)
    {

        // test 1d
        int test_n = 10;
        auto net_cpu = mNet1dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<float> E_cpu_arr = {};
        std::vector<tVectorX> dEdx_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(1);
            batch_x.emplace_back(x);
            float E = net_cpu->forward_unnormed(x);
            tVectorX dEdx = net_cpu->calc_grad_wrt_input_unnormed(x);

            E_cpu_arr.emplace_back(E);
            dEdx_cpu_arr.emplace_back(dEdx);
        }

        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet1dLstGPU[0];

        net_gpu->forward_unnormed_energy_grad_batch(batch_x);

        cTimeUtil::Begin("gpu");
        std::tuple<std::vector<float>, std::vector<tVectorXf>> ret =
            net_gpu->forward_unnormed_energy_grad_batch(batch_x);
        _FLOAT gpu_elasped = cTimeUtil::End("gpu");

        std::vector<float> E_gpu = std::get<0>(ret);
        std::vector<tVectorXf> dEdx_gpu = std::get<1>(ret);

        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < test_n; i++)
        {
            printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
        }
        for (int i = 0; i < test_n; i++)
        {
            printf("[%d] dEdx cpu %.2f gpu %.2f\n", i, dEdx_cpu_arr[i][0],
                   dEdx_gpu[i][0]);
        }
        printf("[1d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
    }
    else
    {

        // test 2d
        int test_n = 10;
        auto net_cpu = mNet2dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<float> E_cpu_arr = {};
        std::vector<tVectorX> dEdx_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(2);
            batch_x.emplace_back(x);
            float E = net_cpu->forward_unnormed(x);
            tVectorX dEdx = net_cpu->calc_grad_wrt_input_unnormed(x);

            E_cpu_arr.emplace_back(E);
            dEdx_cpu_arr.emplace_back(dEdx);
        }

        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet2dLstGPU[0];

        net_gpu->forward_unnormed_energy_grad_batch(batch_x);

        cTimeUtil::Begin("gpu");
        std::tuple<std::vector<float>, std::vector<tVectorXf>> ret =
            net_gpu->forward_unnormed_energy_grad_batch(batch_x);
        _FLOAT gpu_elasped = cTimeUtil::End("gpu");

        std::vector<float> E_gpu = std::get<0>(ret);
        std::vector<tVectorXf> dEdx_gpu = std::get<1>(ret);

        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < test_n; i++)
        {
            printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
        }
        for (int i = 0; i < test_n; i++)
        {
            printf("[%d] dEdx cpu %.2f %.2f gpu %.2f %.2f\n", i,
                   dEdx_cpu_arr[i][0], dEdx_cpu_arr[i][1], dEdx_gpu[i][0],
                   dEdx_gpu[i][1]);
        }
        printf("[2d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
    }
}
#include "utils/ProfUtil.h"
void energy_grad_hess_verify()
{
    bool test_1d = false;
    int test_n = 1;
    if (test_1d)
    {
        // test 1d
        auto net_cpu = mNet1dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<float> E_cpu_arr = {};
        std::vector<tVectorX> dEdx_cpu_arr = {};
        std::vector<tMatrixX> dE2dx2_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(1);
            batch_x.emplace_back(x);
            float E = net_cpu->forward_unnormed(x);
            tVectorX dEdx = net_cpu->calc_grad_wrt_input_unnormed(x);
            tMatrixX dE2dx2 = net_cpu->calc_hess_wrt_input_unnormed(x);

            E_cpu_arr.emplace_back(E);
            dEdx_cpu_arr.emplace_back(dEdx);
            dE2dx2_cpu_arr.emplace_back(dE2dx2);
        }

        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet1dLstGPU[0];

        net_gpu->AdjustGPUBuffer(batch_x.size());

        cTimeUtil::Begin("gpu");

        std::vector<float> E_gpu(test_n);
        std::vector<tVectorXf> dEdx_gpu(test_n, tVectorXf::Zero(1));
        std::vector<tMatrixXf> dE2dx2_gpu(test_n, tMatrixXf::Zero(1, 1));
        net_gpu->forward_unnormed_energy_grad_hess_batch(batch_x, E_gpu,
                                                         dEdx_gpu, dE2dx2_gpu);
        _FLOAT gpu_elasped = cTimeUtil::End("gpu");

        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < test_n; i++)
        {
            printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
        }
        for (int i = 0; i < test_n; i++)
        {
            printf("[%d] dEdx cpu %.2f gpu %.2f\n", i, dEdx_cpu_arr[i][0],
                   dEdx_gpu[i][0]);
        }
        for (int i = 0; i < test_n; i++)
        {
            printf("[%d] dE2dx2 cpu %.2f gpu %.2f\n", i,
                   dE2dx2_cpu_arr[i](0, 0), dE2dx2_gpu[i](0, 0));
        }
        printf("[1d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
    }
    else
    {

        // test 2d
        int test_n = 400000;
        auto net_cpu = mNet2dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<float> E_cpu_arr = {};
        std::vector<tVectorX> dEdx_cpu_arr = {};
        std::vector<tMatrixX> dE2dx2_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        int print_num = std::min(test_n, 2);
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(2);
            batch_x.emplace_back(x);
            if (i < print_num)
            {
                float E = net_cpu->forward_unnormed(x);
                tVectorX dEdx = net_cpu->calc_grad_wrt_input_unnormed(x);
                tMatrix2 dE2dx2 = net_cpu->calc_hess_wrt_input_unnormed(x);

                E_cpu_arr.emplace_back(E);
                dEdx_cpu_arr.emplace_back(dEdx);
                dE2dx2_cpu_arr.emplace_back(dE2dx2);
            }
        }
        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet2dLstGPU[0];
        net_gpu->AdjustGPUBuffer(batch_x.size());

        // net_gpu->forward_unnormed_energy_grad_hess_batch(batch_x);

        std::vector<float> E_gpu(test_n);
        std::vector<tVectorXf> dEdx_gpu(test_n, tVectorXf::Zero(2));
        std::vector<tMatrixXf> dE2dx2_gpu(test_n, tMatrixXf::Zero(2, 2));

        cTimeUtil::Begin("gpu");
        cProfUtil::Begin("gpu");
        net_gpu->forward_unnormed_energy_grad_hess_batch(batch_x, E_gpu,
                                                         dEdx_gpu, dE2dx2_gpu);
        cProfUtil::End("gpu");

        _FLOAT gpu_elasped = cTimeUtil::End("gpu");

        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < print_num; i++)
        {
            printf("[%d] E cpu %.3f gpu %.3f\n", i, E_cpu_arr[i], E_gpu[i]);
        }
        for (int i = 0; i < print_num; i++)
        {
            printf("[%d] dEdx cpu %.3f %.3f gpu %.3f %.3f\n", i,
                   dEdx_cpu_arr[i][0], dEdx_cpu_arr[i][1], dEdx_gpu[i][0],
                   dEdx_gpu[i][1]);
        }
        for (int i = 0; i < print_num; i++)
        {
            printf("[%d] dE2dx2 cpu\n", i);
            std::cout << dE2dx2_cpu_arr[i] << std::endl;
            printf("dE2dx2 gpu\n");
            std::cout << dE2dx2_gpu[i] << std::endl;
            // printf("[%d] dE2dx2 cpu %.3f %.3f gpu %.3f %.3f\n", i,
            //        dEdx_cpu_arr[i][0], dEdx_cpu_arr[i][1], dEdx_gpu[i][0],
            //        dEdx_gpu[i][1]);
        }
        printf("[2d] cpu %.3f gpu %.3f\n", cpu_elasped, gpu_elasped);
        std::cout << cProfUtil::GetTreeDesc("gpu");
    }
}
