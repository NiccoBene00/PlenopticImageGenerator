#pragma once

#include <string>

#include "utils/types.hpp"
#include "utils/JSONUtils.hpp"

struct DisplaySpec {
	int resolutionX;
	int resolutionY;
	float width_mm;
	float height_mm;
	float pixelSize_mm;
};

struct MLASpec {
	std::string type;

	uint16_t countX;
	uint16_t countY;
	uint32_t totalLenses;

	float pitch_mm;
	float focalLength_mm;
	float displayDistance_mm;
};

struct SystemSpec {
	std::string name;
	DisplaySpec display;
	MLASpec mla;
};

template<>
struct JsonLoader<SystemSpec> {
	static std::optional<SystemSpec> load(const nlohmann::json& json) {
		SystemSpec spec;

		spec.name = getRequiredString(json, "NAME");

		nlohmann::json displayJson = getRequiredObject(json, "DISPLAY");
		spec.display.resolutionX = getRequiredInt(displayJson, "RESOLUTION_X");
		spec.display.resolutionY = getRequiredInt(displayJson, "RESOLUTION_Y");
		spec.display.width_mm = getRequiredFloat(displayJson, "WIDTH_MM");
		spec.display.height_mm = getRequiredFloat(displayJson, "HEIGHT_MM");
		const float pixelSizeX = spec.display.width_mm / spec.display.resolutionX;
		const float pixelSizeY = spec.display.width_mm / spec.display.resolutionX;
		spec.display.pixelSize_mm = (pixelSizeX + pixelSizeY) / 2.f;

		nlohmann::json mlaJson = getRequiredObject(json, "MICROLENS_ARRAY");
		spec.mla.type = getRequiredString(mlaJson, "TYPE");
		spec.mla.pitch_mm = getRequiredFloat(mlaJson, "PITCH_MM");
		spec.mla.focalLength_mm = getRequiredFloat(mlaJson, "FOCAL_LENGTH_MM");
		spec.mla.displayDistance_mm = getRequiredFloat(mlaJson, "MLA_DISTANCE_MM");
		spec.mla.countX = static_cast<int>(std::round(spec.display.width_mm / spec.mla.pitch_mm));
		spec.mla.countY = static_cast<int>(std::round(spec.display.height_mm / spec.mla.pitch_mm));
		spec.mla.totalLenses = spec.mla.countX * spec.mla.countY;

		return spec;
	}
};


inline bool operator==(const DisplaySpec& a, const DisplaySpec& b)
{
	return a.resolutionX == b.resolutionX &&
		a.resolutionY == b.resolutionY &&
		a.width_mm == b.width_mm &&
		a.height_mm == b.height_mm &&
		a.pixelSize_mm == b.pixelSize_mm;
}

inline bool operator!=(const DisplaySpec& a, const DisplaySpec& b)
{
	return !(a == b);
}

inline bool operator==(const MLASpec& a, const MLASpec& b)
{
	return a.type == b.type &&
		a.countX == b.countX &&
		a.countY == b.countY &&
		a.totalLenses == b.totalLenses &&
		a.pitch_mm == b.pitch_mm &&
		a.focalLength_mm == b.focalLength_mm &&
		a.displayDistance_mm == b.displayDistance_mm;
}

inline bool operator!=(const MLASpec& a, const MLASpec& b)
{
	return !(a == b);
}

inline bool operator==(const SystemSpec& a, const SystemSpec& b)
{
	return a.name == b.name &&
		a.display == b.display &&
		a.mla == b.mla;
}

inline bool operator!=(const SystemSpec& a, const SystemSpec& b)
{
	return !(a == b);
}



