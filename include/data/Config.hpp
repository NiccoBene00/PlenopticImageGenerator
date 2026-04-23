#pragma once

#include "utils/JSONUtils.hpp"

//###=== STRUCT/ENUM DEFINITION


enum class PointCloudMode {
	REAL,		// all points in front of the display
	VIRTUAL,	// all points behind the display
	MLA,		// Points centered around the MLA position
	CDP			// Points centered around CDP (center depth plane) position
};

inline const std::array<std::pair<std::string, PointCloudMode>, 4>
PointCloudModeStringMap{ {
	{"REAL", PointCloudMode::REAL},
	{"VIRTUAL", PointCloudMode::VIRTUAL},
	{"MLA", PointCloudMode::MLA},
	{"CDP", PointCloudMode::CDP},
}};

inline constexpr EnumParser PointCloudModeParser{ PointCloudModeStringMap };

struct Config {
	std::string configName;
	int superResolutionFactor; 		// Preprocessing for densifying the point cloud
	int crackFilteringKernel;	// Size of the interpolation kernel to inpaint crack artifacts
	PointCloudMode pointCloudMode;	// Where the point cloud is located with respected to the display
	bool entirePipelineGPU; 
};

inline bool operator==(const Config& a, const Config& b)
{
	return a.configName == b.configName &&
		a.superResolutionFactor == b.superResolutionFactor &&
		a.crackFilteringKernel == b.crackFilteringKernel &&
		a.entirePipelineGPU == b.entirePipelineGPU &&
		a.pointCloudMode == b.pointCloudMode;
}

inline bool operator!=(const Config& a, const Config& b)
{
	return !(a == b);
}

//###=== FUNCTION DEFINITION

template<>
struct JsonLoader<Config> {
	static std::optional<Config> load(const nlohmann::json& json) {
		Config config;
		config.configName = getRequiredString(json, "NAME");
		config.superResolutionFactor = getRequiredInt(json, "SUPER_RESOLUTION_FACTOR");
		config.crackFilteringKernel = getRequiredInt(json, "CRACK_FILTERING_KERNEL");
		config.entirePipelineGPU = json["ENTIRE_PIPELINE_GPU"].get<bool>();

		// Use the generic parser
		std::optional<PointCloudMode> ptCloudMode =
			PointCloudModeParser.parseFromJson(json, "POINT_CLOUD_MODE");
		if (!ptCloudMode)
			return std::nullopt;
		config.pointCloudMode = *ptCloudMode;

		return config;
	}
};