#include "StringUtil.h"

std::vector<std::string> cStringUtil::SplitString(const std::string &s_,
                                                  const std::string &delimiter)
{
    size_t pos = 0;
    std::string token;
    std::string s = s_;
    std::vector<std::string> split_string_array(0);
    while ((pos = s.find(delimiter)) != std::string::npos)
    {
        token = s.substr(0, pos);
        split_string_array.push_back(token);
        // std::cout << token << std::endl;
        s.erase(0, pos + delimiter.length());
    }
    split_string_array.push_back(s);
    return split_string_array;
}

void cStringUtil::RemoveEmptyLine(std::vector<std::string> &cont)
{
    std::vector<std::string>::iterator it = cont.begin();
    while (it != cont.end())
    {
        if ((*it).size() == 0)
        {
            it = cont.erase(it);
            it = cont.begin();
        }
        else
        {
            it++;
        }
    }
}

std::string cStringUtil::Strip(std::string line)
{
    if (line.size() == 0)
        return line;
    std::string cur_line = line;
    while (cur_line[0] == ' ' || cur_line[0] == '\n')
    {
        cur_line = cur_line.substr(1, cur_line.size() - 1);
    }
    while (cur_line[cur_line.size() - 1] == ' ' ||
           cur_line[cur_line.size() - 1] == '\n')
    {
        cur_line = cur_line.substr(0, cur_line.size() - 1);
    }
    return cur_line;
}

void cStringUtil::RemoveCommentLine(std::vector<std::string> &cont,
                                    std::string comment_delimeter /*= "#"*/)
{
    std::vector<std::string>::iterator it = cont.begin();
    while (it != cont.end())
    {
        std::string cur_string = *it;
        cur_string = cStringUtil::Strip(cur_string);
        if (cur_string.find(comment_delimeter) == 0)
        {
            it = cont.erase(it);
            it = cont.begin();
        }
        else
        {
            it++;
        }
    }
}

std::string
cStringUtil::ConcatenateString(const std::vector<std::string> &str_vec,
                               std::string delimiter)
{
    return cStringUtil::ConcatenateString(str_vec.begin(), str_vec.end(),
                                          delimiter);
}
#include <algorithm>

std::string
cStringUtil::ConcatenateString(const std::vector<std::string>::const_iterator & begin,
                               const std::vector<std::string>::const_iterator & end,
                               std::string delimiter)
{
    std::string tmp = "";
    auto cur = begin;
    while (cur != end)
    {
        if ((*cur).size() != 0)
        {
            if (tmp.size() != 0)
            {
                tmp += delimiter;
            }
            tmp += *cur;
        }
        cur++;
    }
    return tmp;
}
#include <regex>
std::string cStringUtil::Replace(std::string old_string, std::string tar_str,
                                 std::string new_str)
{
    auto result = std::regex_replace(old_string, std::regex(tar_str), new_str);
    return result;
}