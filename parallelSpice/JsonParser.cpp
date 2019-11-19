/*
 * JsonParser.cpp - a simple JSON parser to load the circuits
 * author: Justin Tremblay, 2019
 */
#include "JsonParser.h"
#include <fstream>
#include "include/rapidjson/istreamwrapper.h"
#include "dbg.h"

// Initialize the parser with a filepath
JsonParser::JsonParser(const std::string& file_path) {
    m_doc = {}; // initialize the JSON document
    auto ifs = std::ifstream{file_path}; // load file into a stream
    rapidjson::IStreamWrapper isw{ifs}; // wrap around the stream in order to parse
    m_doc.ParseStream(isw); // parse the stream
}

// Parse the loaded document
_Check_return_ bool JsonParser::parse(size_t& dim, size_t& timepoints, cuda_mem::grid<double>& a_grid, cuda_mem::grid<double>& l_grid, cuda_mem::grid<double>& bs_grid, cuda_mem::grid<double>& xs_grid) {
    // Make sure the document properly parsed
    if (m_doc.HasParseError()) {
        dbg(m_doc.GetParseError());
        return false;
    }
    
    // Get the dimension
    if (!m_doc.HasMember("dimension")) return false;
    dim = m_doc["dimension"].GetInt();

    // Get the number of time points
    if (!m_doc.HasMember("timepoints")) return false;
    timepoints = m_doc["timepoints"].GetInt();
    
    // parse the A matrix
    if (!m_doc.HasMember("A")) return false;
    dbg("Parsing A matrix...\n");
    const auto& a_doc = m_doc["A"];
    if (!parse_grid(a_doc, a_grid)) return false;
    
    // parse the L matrix
    if (!m_doc.HasMember("L")) return false;
    dbg("Parsing L matrix...\n");
    const auto& l_doc = m_doc["L"];
    if (!parse_grid(l_doc, l_grid)) return false;
    
    // parse the Bs Vector
    if (!m_doc.HasMember("Bs")) return false;
    dbg("Parsing Bs vector...\n");
    const auto& bs_doc = m_doc["Bs"];
    if (!parse_grid(bs_doc, bs_grid)) return false;
    
    // parse the Xs Vector
    if (!m_doc.HasMember("Xs")) return false;
    dbg("Parsing Xs vector...\n");
    const auto& xs_doc = m_doc["Xs"];
    if (!parse_grid(xs_doc, xs_grid)) return false;

    // Successfully reached the end
    return true;
}

_Check_return_ bool JsonParser::parse_grid(const rapidjson::Value& doc, cuda_mem::grid<double> grid) {
    grid = cuda_mem::grid<double>();
    if (doc.IsArray()) {
        // Matrices are represented as arrays or arrays
        for (const auto& arr : doc.GetArray()) {
            grid.emplace_back(std::vector<double>());
            for (const auto& f : arr.GetArray()) {
                grid.back().emplace_back(f.GetDouble());
            }
        }
    } else {
        return false;
    }
    return true;
}
