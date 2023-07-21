#ifndef NO_IMGUI
#include "utils/ImGuiUtil.h"

void cImGuiUtil::SliderScalar(std::string name, _FLOAT *data, _FLOAT min,
                              _FLOAT max, std::string format)
{
    if constexpr (std::is_same<float, _FLOAT>::value)
    {
        // float
        ImGui::SliderScalar(name.c_str(), ImGuiDataType_Float, data, &min, &max,
                            format.c_str());
    }
    else
    {
        // double
        ImGui::SliderScalar(name.c_str(), ImGuiDataType_Double, data, &min,
                            &max, format.c_str());
    }
}
void cImGuiUtil::SliderScalar3(std::string name, _FLOAT *data, _FLOAT min,
                               _FLOAT max, std::string format)
{
    if constexpr (std::is_same<float, _FLOAT>::value)
    {
        // float
        ImGui::SliderScalarN(name.c_str(), ImGuiDataType_Float, data, 3, &min,
                             &max, format.c_str());
    }
    else
    {
        // double
        ImGui::SliderScalarN(name.c_str(), ImGuiDataType_Double, data, 3, &min,
                             &max, format.c_str());
    }
}
void cImGuiUtil::DragScalar(std::string name, _FLOAT *data, _FLOAT speed,
                            _FLOAT min, _FLOAT max, std::string format)
{
    if constexpr (std::is_same<float, _FLOAT>::value)
    {
        // float
        ImGui::DragScalar(name.c_str(), ImGuiDataType_Float, data, speed, &min,
                          &max, format.c_str());
    }
    else
    {
        // double
        ImGui::DragScalar(name.c_str(), ImGuiDataType_Double, data, speed, &min,
                          &max, format.c_str());
    }
}
void cImGuiUtil::DragScalar3(std::string name, _FLOAT *data, _FLOAT speed,
                             _FLOAT min, _FLOAT max, std::string format)
{
    if constexpr (std::is_same<float, _FLOAT>::value)
    {
        // float
        ImGui::DragScalarN(name.c_str(), ImGuiDataType_Float, data, 3, speed,
                           &min, &max, format.c_str());
    }
    else
    {
        // double
        ImGui::DragScalarN(name.c_str(), ImGuiDataType_Double, data, 3, speed,
                           &min, &max, format.c_str());
    }
}

void cImGuiUtil::TextSeperator(const std::string &text)
{

    float text_width = ImGui::CalcTextSize(text.c_str()).x;
    float avail_width = ImGui::GetContentRegionAvailWidth();
    float separator_start_pos = (avail_width - text_width) * 0.5f;

    ImGui::Separator();
    ImGui::SetCursorPosX(separator_start_pos);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
    ImGui::Separator();
}
#include "utils/LogUtil.h"
void cImGuiUtil::TextCentered(const std::string &text)
{
    // Calculate the size of the text
    ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    // Calculate the position for the centered text
    float text_pos_x =
        (ImGui::GetContentRegionAvailWidth() - text_size.x) * 0.5f;
    // Set the cursor position before rendering the text
    ImGui::SetCursorPosX(text_pos_x);
    ImGui::TextUnformatted(text.c_str());
}
#endif