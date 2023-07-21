#include "ProfUtil.h"
#include "utils/LogUtil.h"
#include "utils/StringUtil.h"
#include "utils/TimeUtil.hpp"

// ================= Prof Node ===================

struct tProfNode
{
    tProfNode(std::string name);
    bool StartChild(std::string);
    bool EndChild(std::string);
    float GetCost_us() const;
    bool IsValidRecord() const;
    bool IsLeafNode() const;

    std::string mName;
    cTimePoint mCurSt, mCurEd;
    std::vector<float> mHistoryElapsed_us; // support open the profiling node
                                           // many times(micro second)
    std::vector<tProfNodePtr> mChildArray;
};
tProfNode::tProfNode(std::string name)
{
    mName = name;
    mHistoryElapsed_us.clear();
    mChildArray.clear();
}

bool tProfNode::StartChild(std::string name)
{
    auto layers = cStringUtil::SplitString(name, "/");
    SIM_ASSERT(layers[0] == this->mName);

    if (layers.size() == 2)
    {
        bool exist = false;
        tProfNodePtr child = nullptr;
        for (auto &x : mChildArray)
        {
            if (x->mName == layers[1])
            {
                exist = true;
                child = x;
                break;
            }
        }
        // insert
        if (child == nullptr)
        {
            child = std::make_shared<tProfNode>(layers[1]);
            mChildArray.push_back(child);
        }

        child->StartChild(layers[1]);
    }
    else if (layers.size() > 2)
    {
        // insert into next level
        for (auto &x : mChildArray)
        {
            if (layers[1] == x->mName)
            {
                x->StartChild(cStringUtil::ConcatenateString(
                    layers.begin() + 1, layers.end(), "/"));
            }
        }
    }
    else if (layers.size() == 1)
    {
        mCurSt = cTimeUtil::GetCurrentTime_chrono();
    }
    return true;
}
bool tProfNode::EndChild(std::string name)
{
    // printf("[debug] begin to end child %s\n", name.c_str());
    auto layers = cStringUtil::SplitString(name, "/");
    SIM_ASSERT(layers[0] == this->mName);

    if (layers.size() > 1)
    {
        bool updated = false;
        for (int i = 0; i < mChildArray.size(); i++)
        {
            auto &x = mChildArray[i];
            if (layers[1] == x->mName)
            {
                updated = true;

                x->EndChild(cStringUtil::ConcatenateString(layers.begin() + 1,
                                                           layers.end(), "/"));
            }
        }
        SIM_ASSERT(updated = true);
    }
    else
    {
        // myself end

        mCurEd = cTimeUtil::GetCurrentTime_chrono();
        mHistoryElapsed_us.push_back(
            cTimeUtil::CalcTimeElaspedus(mCurSt, mCurEd));
    }

    return true;
}
#include <iostream>
#include <numeric>
float tProfNode::GetCost_us() const
{
    return std::accumulate(mHistoryElapsed_us.begin(), mHistoryElapsed_us.end(),
                           0.0);
}
bool tProfNode::IsValidRecord() const
{
    return cTimeUtil::CalcTimeElaspedus(mCurEd, mCurSt) > 0;
}

bool tProfNode::IsLeafNode() const { return mChildArray.size() == 0; }

std::vector<tProfNodePtr> cProfUtil::mRootArray = {};

// ================= Prof Util ===================

void cProfUtil::Begin(std::string name)
{
    // 1. get root name
    auto layers = cStringUtil::SplitString(name, "/");
    std::string root_name = layers[0];
    bool visited = false;
    for (auto &x : mRootArray)
    {
        if (x->mName == root_name)
        {
            x->StartChild(name);
            visited = true;
        }
    }
    // create new root
    if (visited == false)
    {
        auto node = std::make_shared<tProfNode>(root_name);
        mRootArray.push_back(node);
        node->StartChild(name);
    }
}
void cProfUtil::End(std::string name)
{
    auto layers = cStringUtil::SplitString(name, "/");
    std::string root_name = layers[0];
    bool visited = false;
    for (int i = 0; i < mRootArray.size(); i++)
    {
        auto x = mRootArray[i];
        if (x->mName == root_name)
        {
            // printf("[debug] end root %s\n", x->mName.c_str());
            x->EndChild(name);
            visited = true;
        }
    }
    SIM_ASSERT(visited);
}

// void cProfUtil::Clear() { mRootArray.clear(); }
#include <iostream>
std::string PrintProfTree(const std::string &prefix, const tProfNodePtr node,
                          bool is_first, float total_cost_us)
{
    std::string ret_str = "";
    if (node != nullptr)
    {
        ret_str += prefix;

        // ret_str += (is_first ? "\u251C\u2500\u2500" : "\u2514\u2500\u2500");
        ret_str += (is_first ? "|--" : "|--");

        // print the value of the node
        char output[200] = {};
        if (total_cost_us > 1e3)
        {

            sprintf(output, "%s %.1fms(%.1f%%)\n", node->mName.c_str(),
                    node->GetCost_us() / 1e3,
                    node->GetCost_us() / total_cost_us * 100);
        }
        else
        {
            sprintf(output, "%s %.1fus(%.1f%%)\n", node->mName.c_str(),
                    node->GetCost_us(),
                    node->GetCost_us() / total_cost_us * 100);
        }
        ret_str += std::string(output);
        // enter the next tree level - left and right branch
        for (int i = 0; i < node->mChildArray.size(); i++)
        {
            ret_str +=
                PrintProfTree(prefix + (is_first ? "|--" : "   "),
                              node->mChildArray[i], i == 0, total_cost_us);
        }
    }
    return ret_str;
}
std::string cProfUtil::GetTreeDesc(std::string name)
{
    std::string output = "";
    for (auto &x : mRootArray)
    {
        if (x->mName == name)
        {
            output = PrintProfTree("", x, false, x->GetCost_us());
            break;
        }
    }
    return output;
}

void cProfUtil::ClearAll() { mRootArray.clear(); }

void cProfUtil::ClearRoot(std::string name)
{
    auto it = mRootArray.begin();
    while (it != mRootArray.end())
    {
        if (name == (*it)->mName)
        {
            mRootArray.erase(it);
            return;
        }
        it++;
    }
}

float cProfUtil::GetElapsedTime(std::string name)
{
    auto layers = cStringUtil::SplitString(name, "/");
    std::string root_name = layers[0];
    tProfNodePtr cur_node = nullptr;
    for (int i = 0; i < mRootArray.size(); i++)
    {
        auto x = mRootArray[i];
        if (x->mName == root_name)
        {
            cur_node = x;
            layers.erase(layers.begin());
        }
    }

    // only one layer
    while (layers.size() > 0)
    {
        std::string tar_name = layers[0];
        for (auto &x : cur_node->mChildArray)
        {
            if (x->mName == tar_name)
            {
                cur_node = x;
                layers.erase(layers.begin());
                break;
            }
        }
    }

    if (cur_node == nullptr)
        return 0;
    else
        return cur_node->GetCost_us();
}