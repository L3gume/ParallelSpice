/*
 * JsonParser.cpp - a simple JSON parser to load the circuits
 * author: Justin Tremblay, 2019
 */
#include <fstream>
#include "JsonParser.h"
#include "include/rapidjson/istreamwrapper.h"
#include "dbg.h"
#include "cuda_utils.h"

using namespace json;

// Initialize the parser with a filepath
JsonParser::JsonParser(const std::string& file_path) {
    m_doc = {}; // initialize the JSON document
    if (cuda_utils::file_exists(file_path.c_str())) {
        auto ifs = std::ifstream{ file_path }; // load file into a stream
        rapidjson::IStreamWrapper isw{ ifs }; // wrap around the stream in order to parse
        m_doc.ParseStream(isw); // parse the stream
    } else {
        std::cerr << "Entered file does not exists! Exiting..." << '\n';
        exit(1);
    }
}

// Parse the loaded document
_Check_return_ bool JsonParser::parse(parse_result& result) {
    // Make sure the document properly parsed
    if (m_doc.HasParseError()) {
        dbg(m_doc.GetParseError());
        return false;
    }
    
    // Get the dimension
    if (!m_doc.HasMember("N")) return false;
    result.n = m_doc["N"].GetInt();
    if (!m_doc.HasMember("M")) return false;
    result.m = m_doc["M"].GetInt();

    // Get the number of time points
    if (!m_doc.HasMember("timepoints")) return false;
    result.timepoints = m_doc["timepoints"].GetInt();
    
    // parse the A matrix
    if (!m_doc.HasMember("A")) return false;
    dbg("Parsing A matrix...\n");
    const auto& a_doc = m_doc["A"];
    if (!parse_grid(a_doc, result.A)) return false;
    
    // parse the L matrix
    if (!m_doc.HasMember("J")) return false;
    dbg("Parsing J vector...\n");
    const auto& j_doc = m_doc["J"];
    if (!parse_vector(j_doc, result.J)) return false;
    
    // parse the Y Vector
    if (!m_doc.HasMember("Y")) return false;
    dbg("Parsing Y vector...\n");
    const auto& y_doc = m_doc["Y"];
    if (!parse_vector(y_doc, result.Y)) return false;
    
    // parse the E Vector
    if (!m_doc.HasMember("E")) return false;
    dbg("Parsing E vector...\n");
    const auto& e_doc = m_doc["E"];
    if (!parse_vector(e_doc, result.E)) return false;
    
    // parse the F Vector
    if (!m_doc.HasMember("F")) return false;
    dbg("Parsing F vector...\n");
    const auto& f_doc = m_doc["F"];
    if (!parse_vector(f_doc, result.F)) return false;

    // Successfully reached the end
    return true;
}

_Check_return_ bool JsonParser::parse_grid(const rapidjson::Value& doc, cuda_mem::grid<double>& grid) {
    grid = cuda_mem::grid<double>();
    if (doc.IsArray()) {
        // Matrices are represented as arrays or arrays
        for (const auto& arr : doc.GetArray()) {
            grid.emplace_back(std::vector<double>());
            for (const auto& d : arr.GetArray()) {
                grid.back().emplace_back(d.GetDouble());
            }
        }
    } else {
        return false;
    }
    return true;
}

_Check_return_ bool JsonParser::parse_vector(const rapidjson::Value& doc, std::vector<double>& vec) {
    vec = std::vector<double>();
    if (doc.IsArray()) {
        for (const auto& d : doc.GetArray()) {
            vec.emplace_back(d.GetDouble());
        }
    } else {
        return false;
    }
    return true;
}
