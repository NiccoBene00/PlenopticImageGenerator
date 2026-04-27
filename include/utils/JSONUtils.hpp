#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <iostream>

namespace fs = std::filesystem;

inline void validateJsonExtension(const std::string& path, const std::string& desc)
{
	if (std::filesystem::path(path).extension() != ".json")
	{
		throw std::invalid_argument("Error: " + desc + " must be a .json file");
	}
}

static std::optional<std::string> resolvePath(const std::string& filePath) {
	// This function determines the path to the root of the project and not the current folder of the executable.
	// It also solves for slash or backslash preference (Win vs Linux)
	fs::path p(filePath);

	if (p.is_absolute()) {
		if (fs::exists(p)) return p.make_preferred().string();
		else {
			std::cerr << "Error: File does not exist at absolute path " << p << std::endl;
			return std::nullopt;
		}
	}

	// Relative: first try current working directory
	if (fs::exists(p)) return p.make_preferred().string();

	fs::path root(PROJECT_ROOT_DIR);
	fs::path combined = root / p;
	if (fs::exists(combined)) return combined.make_preferred().string();

	std::cerr << "Error: File not found at " << p << " relative to cwd or project root " << PROJECT_ROOT_DIR << std::endl;
	return std::nullopt;
}

static std::optional<nlohmann::json> jsonFromFile(const std::string& filePath) {
	std::ifstream inputFile(filePath);
	std::cout << "## Loading JSON from file: " << filePath << std::endl;
	if (!inputFile) {
		std::cerr << "Error: Could not open file " << filePath << std::endl;
		return std::nullopt;
	}
	nlohmann::json jsonData;
	try {
		inputFile >> jsonData;
	}
	catch (const nlohmann::json::parse_error& e) {
		std::cerr << "Error: Failed to parse JSON from file " << filePath << ": " << e.what() << std::endl;
		return std::nullopt;
	}
	return jsonData;
}

template<typename T>
struct JsonLoader {
	// This struct template is specialized for each parameter that needs to be loaded from json
	// DataParameters, SystemSpec
	static std::optional<T> load(const nlohmann::json& json);
};

template<typename T>
std::optional<T> fromFile(const std::string& filePath) {
	std::optional<nlohmann::json> jsonData = jsonFromFile(filePath);
	if (!jsonData || jsonData->is_null()) {
		std::cerr << "Error: Failed to load JSON from file " << filePath << std::endl;
		return std::nullopt;
	}

	return JsonLoader<T>::load(*jsonData);
}

template<typename EnumType, size_t N>
class EnumParser {
private:
	const std::array<std::pair<std::string, EnumType>, N>& stringMap;

public:
	constexpr EnumParser(const std::array<std::pair<std::string, EnumType>, N>& map)
		: stringMap(map) {
	}

	std::string toString(EnumType value) const {
		for (const auto& [name, val] : stringMap)
			if (val == value)
				return name;
		return "UNKNOWN";
	}

	std::optional<EnumType> fromString(const std::string& s) const {
		for (const auto& [name, val] : stringMap)
			if (name == s)
				return val;
		return std::nullopt;
	}

	std::optional<EnumType> parseFromJson(const nlohmann::json& json, const std::string& key) const {
		if (!json.contains(key) || !json[key].is_string()) {
			std::cerr << "Error: Invalid or missing " << key << std::endl;
			throw std::runtime_error(key + " missing or invalid");
		}

		std::optional<EnumType> result = fromString(json[key].get<std::string>());
		if (!result) {
			std::cerr << "Error: Unknown " << key << ". Valid values:\n";
			for (const auto& [k, _] : stringMap)
				std::cerr << "  " << k << "\n";
		}

		return result;
	}
};

// The following are data validation functions obtained from json
inline const int getRequiredInt(const nlohmann::json& j, const std::string& key) {
	if (!j.contains(key) || !j[key].is_number_integer()) {
		std::cerr << "Error: Invalid or missing " << key << std::endl;
		throw std::runtime_error(key + " missing or invalid");
	}
	return j[key].get<int>();
}

inline const float getRequiredFloat(const nlohmann::json& j, const std::string& key) {
	if (!j.contains(key) || !j[key].is_number()) {
		std::cerr << "Error: Invalid or missing " << key << std::endl;
		throw std::runtime_error(key + " missing or invalid");
	}
	return j[key].get<float>();
}


inline const bool getRequiredBool(const nlohmann::json& j, const std::string& key) {
	if (!j.contains(key) || !j[key].is_boolean()) {
		std::cerr << "Error: Invalid or missing " << key << std::endl;
		throw std::runtime_error(key + " missing or invalid");
	}
	return j[key].get<bool>();
}

inline const std::string getRequiredString(const nlohmann::json& j, const std::string& key) {
	if (!j.contains(key) || !j[key].is_string()) {
		std::cerr << "Error: Invalid or missing " << key << std::endl;
		throw std::runtime_error(key + " missing or invalid");
	}
	return j[key].get<std::string>();
}

inline const nlohmann::json& getRequiredObject(const nlohmann::json& j, const std::string& key) {
	if (!j.contains(key) || !j[key].is_object()) {
		std::cerr << "Error: Invalid or missing " << key << std::endl;
		throw std::runtime_error(key + " missing or invalid");
	}
	return j[key];
}
