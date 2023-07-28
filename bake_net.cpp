
#include "FullConnectedNetSingleScalar.h"
#include "FullConnectedNetSingleScalarGPU.h"
#include "utils/JsonUtil.h"
#include "utils/json/json.h"
#include "utils/LogUtil.h"
#include <fstream>

extern std::vector<cFCNetworkSingleScalarPtr> mNet1dLstCPU, mNet2dLstCPU;
extern std::vector<cFCNetworkSingleScalarGPUPtr> mNet1dLstGPU, mNet2dLstGPU;
void Bake1DNet(cFCNetworkSingleScalarPtr net, int samples,
               std::vector<float> &x_samples, std::vector<float> &e_arr,
               std::vector<float> &grad_arr, std::vector<float> &hess_arr)
{
    std::cout << "net = " << net << std::endl;
    float min = net->mXRange(0, 0), max = net->mXRange(1, 0);

    e_arr.resize(samples);
    grad_arr.resize(samples);
    hess_arr.resize(samples);

    // Step 1. sample array from min to max by samples
    x_samples.resize(samples);
    float step = (max - min) / samples;
    for (int i = 0; i < samples; ++i)
    {
        x_samples[i] = min + i * step;
    }

    // Assuming e_arr, grad_arr, hess_arr are already defined and have enough
    // space Here they are initialized as vectors of appropriate types for the
    // example

    // Step 2. for each x, call forward_normed and put them into e_arr,
    // grad_arr, hess_arr
    for (int i = 0; i < samples; ++i)
    {
        e_arr[i] = net->forward_unnormed(tVectorX::Ones(1) * x_samples[i]);
        grad_arr[i] = net->calc_grad_wrt_input_unnormed(tVectorX::Ones(1) *
                                                        x_samples[i])[0];
        hess_arr[i] = net->calc_hess_wrt_input_unnormed(tVectorX::Ones(1) *
                                                        x_samples[i])(0, 0);
    }
}

void Bake2DNet(cFCNetworkSingleScalarGPUPtr net, int N,
               std::vector<tVector2f> &points, std::vector<float> &e_arr,
               std::vector<tVector2f> &grad_arr,
               std::vector<tMatrix2f> &hess_arr)
{
    // int N = 1000; // number of samples
    // float xmin = -1, xmax = 1, ymin = -1, ymax = 1;
    float xmin = net->mXRange(0, 0);
    float xmax = net->mXRange(1, 0);
    float ymin = net->mXRange(0, 1);
    float ymax = net->mXRange(1, 1);

    tEigenArr<tVectorX> pts_arr;
    pts_arr.resize(N * N);
    e_arr.resize(N * N);
    grad_arr.resize(N * N);
    hess_arr.resize(N * N);

    float xstep = (xmax - xmin) / (N - 1);
    float ystep = (ymax - ymin) / (N - 1);

    printf("begin get x\n");
    for (int i = 0; i < N; ++i)
    {
        // printf("%d\n", i);
        for (int j = 0; j < N; ++j)
        {
            float x = xmin + i * xstep;
            float y = ymin + j * ystep;
            pts_arr[i * N + j] = tVector2(x, y);
        }
    }

    {
        printf("begin allo\n");
        std::vector<float> tmp1;
        std::vector<tVectorXf> tmp2;
        std::vector<tMatrixXf> tmp3;
        tmp1.resize(pts_arr.size());
        tmp2.resize(pts_arr.size(), tVectorXf::Zero(2));
        tmp3.resize(pts_arr.size(), tMatrixXf::Zero(2, 2));

        printf("begin forward\n");
        net->forward_unnormed_energy_grad_hess_batch(pts_arr, tmp1, tmp2, tmp3);

        printf("begin transform\n");
        points.resize(N * N);
        e_arr.resize(N * N);
        grad_arr.resize(N * N);
        hess_arr.resize(N * N);

        for (int i = 0; i < N * N; i++)
        {
            points[i] = pts_arr[i].cast<float>();
            e_arr[i] = tmp1[i];
            grad_arr[i] = tmp2[i];
            hess_arr[i] = tmp3[i];
        }
    }
}

std::string BakeNet(std::string raw_dnn_path, int samples)
{

    Json::Value root;
    for (int i = 0; i < mNet1dLstCPU.size(); i++)
    {
        printf("begin %d\n", i);
        std::vector<float> x_arr, e_arr, grad_arr, hess_arr;
        Bake1DNet(mNet1dLstCPU[i], samples, x_arr, e_arr, grad_arr, hess_arr);
        printf("succ %d\n", i);
        SIM_INFO("x {} e {} grad {} hess {}", x_arr[0], e_arr[0], grad_arr[0],
                 hess_arr[0]);

        // Add data to JSON
        for (int j = 0; j < samples; j++)
        {
            root["1DNet"][i]["x"].append(x_arr[j]);
            root["1DNet"][i]["e"].append(e_arr[j]);
            root["1DNet"][i]["grad"].append(grad_arr[j]);
            root["1DNet"][i]["hess"].append(hess_arr[j]);
        }
    }

    for (int i = 0; i < mNet2dLstCPU.size(); i++)
    {
        std::vector<tVector2f> x_arr;
        std::vector<float> e_arr;
        std::vector<tVector2f> grad_arr;
        std::vector<tMatrix2f> hess_arr;

        auto net = mNet2dLstGPU[i];
        Bake2DNet(net, samples, x_arr, e_arr, grad_arr, hess_arr);
        SIM_INFO("x {} e {} grad {} hess {}", x_arr[0], e_arr[0], grad_arr[0],
                 hess_arr[0]);

        // Add data to JSON
        OMP_PARALLEL_FOR(OMP_MAX_THREADS)
        for (int j = 0; j < samples * samples; j++)
        {
            root["2DNet"][i]["x"].append(Json::arrayValue);
            root["2DNet"][i]["x"][j].append(x_arr[j].x());
            root["2DNet"][i]["x"][j].append(x_arr[j].y());
            // printf("(%d , %d) x = %.3f %.3f\n", i, j, x_arr[j].x(),
            //        x_arr[j].y());
            root["2DNet"][i]["e"].append(e_arr[j]);

            root["2DNet"][i]["grad"].append(Json::arrayValue);
            root["2DNet"][i]["grad"][j].append(grad_arr[j].x());
            root["2DNet"][i]["grad"][j].append(grad_arr[j].y());

            root["2DNet"][i]["hess"].append(Json::arrayValue);
            root["2DNet"][i]["hess"][j].append(hess_arr[j](0, 0));
            root["2DNet"][i]["hess"][j].append(hess_arr[j](0, 1));
            root["2DNet"][i]["hess"][j].append(hess_arr[j](1, 0));
            root["2DNet"][i]["hess"][j].append(hess_arr[j](1, 1));
        }
    }

    // Write JSON to file
    std::string baked_info_path = "baked_" + raw_dnn_path;
    std::ofstream file_id(baked_info_path);
    file_id << root;
    file_id.close();
    return baked_info_path;
}