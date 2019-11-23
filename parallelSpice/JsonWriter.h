#pragma once
#include "include/rapidjson/document.h"
#include "cuda_mem.h"

namespace json {
	class JsonWriter {
	public:
		explicit JsonWriter(const std::string& input_file_path);
		_Check_return_ bool write(float timestep, const cuda_mem::grid<double>& grid);

	private:
		_Check_return_ bool write_grid(rapidjson::Value& doc, const cuda_mem::grid<double>& grid);
		rapidjson::Value write_vec(const std::vector<double>& vec, const rapidjson::Document::AllocatorType& alloc);

		rapidjson::Document m_doc;
		std::string m_output_file;
	};
}
