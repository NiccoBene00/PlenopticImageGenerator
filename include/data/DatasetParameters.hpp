#pragma once

#include "utils/JSONUtils.hpp"

enum class DepthEncoding {
	FLOAT_METRIC,
	MPEG_DISPARITY
};

inline const std::array<std::pair<std::string, DepthEncoding>, 2>
DepthEncodingStringMap {{
    {"FLOAT_METRIC", DepthEncoding::FLOAT_METRIC},
    {"MPEG_DISPARITY", DepthEncoding::MPEG_DISPARITY}
}};

inline constexpr EnumParser DepthEncodingParser{ DepthEncodingStringMap };

struct DatasetParameters {
	std::string datasetName;
	std::string rgbImagePath;
	std::string depthMapPath;
	DepthEncoding depthEncoding;

	float CAM_FX_px;
	float CAM_FY_px;
	float CAM_PX_px;
	float CAM_PY_px;

	float nearPlane_m = 0.f;
	float farPlane_m = 0.f;
	int nBitsEncoded = 0;
	int depthWidth = 0;
	int depthHeight = 0;
};

template<>
struct JsonLoader<DatasetParameters> {
	static std::optional<DatasetParameters> load(const nlohmann::json& json) {

		DatasetParameters params;
		params.datasetName = getRequiredString(json, "NAME");

		std::string imagePath = getRequiredString(json, "IMAGE_PATH");
		std::optional<std::string> possibleImgPath = resolvePath(imagePath);
		if (!possibleImgPath) {
			return std::nullopt;
		}
		params.rgbImagePath = possibleImgPath.value();

		std::string depthPath = getRequiredString(json, "DEPTH_PATH");
		std::optional<std::string> possibleDepthPath = resolvePath(depthPath);
		if (!possibleDepthPath) {
			return std::nullopt;
		}
		params.depthMapPath = possibleDepthPath.value();

		std::optional<DepthEncoding> depthEncoding = DepthEncodingParser.parseFromJson(json, "DEPTH_ENCODING");
		if (!depthEncoding)
			return std::nullopt;
		params.depthEncoding = depthEncoding.value();

		if (params.depthEncoding == DepthEncoding::MPEG_DISPARITY) {
			params.nearPlane_m = getRequiredFloat(json, "NEAR_PLANE_m");
			params.farPlane_m = getRequiredFloat(json, "FAR_PLANE_m");
			params.nBitsEncoded = getRequiredInt(json, "N_BITS_ENCODED");
			params.depthWidth = getRequiredInt(json, "DEPTH_WIDTH");
			params.depthHeight = getRequiredInt(json, "DEPTH_HEIGHT");
		}

		params.CAM_FX_px = getRequiredFloat(json, "CAMERA_FX_px");
		params.CAM_FY_px = getRequiredFloat(json, "CAMERA_FY_px");
		params.CAM_PX_px = getRequiredFloat(json, "CAMERA_PX_px");
		params.CAM_PY_px = getRequiredFloat(json, "CAMERA_PY_px");

		return params;
	}
};


inline bool operator==(const DatasetParameters& a, const DatasetParameters& b)
{
	return a.datasetName == b.datasetName &&
		a.rgbImagePath == b.rgbImagePath &&
		a.depthMapPath == b.depthMapPath &&
		a.depthEncoding == b.depthEncoding &&
		a.CAM_FX_px == b.CAM_FX_px &&
		a.CAM_FY_px == b.CAM_FY_px &&
		a.CAM_PX_px == b.CAM_PX_px &&
		a.CAM_PY_px == b.CAM_PY_px &&
		a.nearPlane_m == b.nearPlane_m &&
		a.farPlane_m == b.farPlane_m &&
		a.nBitsEncoded == b.nBitsEncoded &&
		a.depthWidth == b.depthWidth &&
		a.depthHeight == b.depthHeight;
}

inline bool operator!=(const DatasetParameters& a, const DatasetParameters& b)
{
	return !(a == b);
}


