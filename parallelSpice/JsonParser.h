/*
 * JsonParser.h - a simple JSON parser to load the circuits
 * author: Justin Tremblay, 2019
 */
#pragma once

#include <string>
#include "include/rapidjson/document.h"
#include "cuda_mem.h"

// Simple implementation of a JSON parser for the project.
// Built with RapidJSON
class JsonParser {
public:
    explicit JsonParser(const std::string& file_path);
    _Check_return_ bool parse(size_t& dim, size_t& timepoints,
                              cuda_mem::grid<double>& a_grid, cuda_mem::grid<double>& l_grid,
                              cuda_mem::grid<double>& bs_grid, cuda_mem::grid<double>& xs_grid);
private:
    _Check_return_ static bool parse_grid(const rapidjson::Value& doc, cuda_mem::grid<double> grid);
    
    rapidjson::Document m_doc;
};
