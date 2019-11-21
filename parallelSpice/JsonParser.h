/*
 * JsonParser.h - a simple JSON parser to load the circuits
 * author: Justin Tremblay, 2019
 */
#pragma once

#include <string>
#include "include/rapidjson/document.h"
#include "cuda_mem.h"

namespace json {

struct parse_result {
    cuda_mem::grid<double> A;
    std::vector<double> J;
    std::vector<double> Y;
    std::vector<double> E;
    std::vector<double> F;

    size_t n;
    size_t m;
};

// Simple implementation of a JSON parser for the project.
// Built with RapidJSON
class JsonParser {
public:
    explicit JsonParser(const std::string& file_path);
    _Check_return_ bool parse(parse_result& result);
private:
    _Check_return_ static bool parse_grid(const rapidjson::Value& doc, cuda_mem::grid<double>& grid);
    _Check_return_ static bool parse_vector(const rapidjson::Value& doc, std::vector<double>& vec);

    rapidjson::Document m_doc;
};

}
