#include <fstream>
#include "JsonWriter.h"
#include "cuda_utils.h"
#include "include/rapidjson/ostreamwrapper.h"
#include "include/rapidjson/prettywriter.h"
#include "include/rapidjson/stringbuffer.h"

using namespace json;

JsonWriter::JsonWriter(const std::string& input_file_path) {
	m_output_file = cuda_utils::make_output_file(input_file_path, "_output", "json");
	m_doc = {};
}

_Check_return_ bool JsonWriter::write(const float timestep, const cuda_mem::grid<double>& grid) {
	m_doc.SetObject();

	m_doc.AddMember(rapidjson::GenericStringRef<char>("timestep"), timestep, m_doc.GetAllocator());
	m_doc.AddMember(rapidjson::GenericStringRef<char>("n_entries"), grid.size(), m_doc.GetAllocator());

	rapidjson::Document entries(&m_doc.GetAllocator());
	entries.SetArray();
	if (!write_grid(entries, grid)) return false;
	m_doc.AddMember("entries", entries, m_doc.GetAllocator());

	std::ofstream ofs(m_output_file);
	rapidjson::OStreamWrapper osw(ofs);
	rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
	m_doc.Accept(writer);
	
	return true;
}

_Check_return_ bool JsonWriter::write_grid(rapidjson::Value& doc, const cuda_mem::grid<double>& grid) {
	if (!doc.IsArray()) return false;

	for (const auto& vec : grid) {
		auto arr = write_vec(vec, m_doc.GetAllocator());
		doc.PushBack(arr, m_doc.GetAllocator());
	}

	return true;
}

rapidjson::Value JsonWriter::write_vec(const std::vector<double>& vec, const rapidjson::Document::AllocatorType& alloc) {
	rapidjson::Value arr(rapidjson::kArrayType);
	for (const auto& d : vec) {
		arr.PushBack(d, m_doc.GetAllocator());
	}
	
	return arr;
}
