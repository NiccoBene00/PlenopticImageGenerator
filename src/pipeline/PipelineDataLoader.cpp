#include "pipeline/PipelineDataLoader.hpp"

cv::Mat PipelineDataLoader::convertMPEGDisparityToFloatMetric(const cv::Mat& disparityMap, const int nBitsEncoded, const float near, const float far)
{
	CV_Assert(nBitsEncoded > 0 && nBitsEncoded <= 16);
	CV_Assert(near > 0 && far > near);

	const float FAR_PLANE_LIMIT_m = 1000.f;
	const float maxEncodedValue = static_cast<float>((1 << nBitsEncoded) - 1);

	// Create output depth map as float
	cv::Mat depthMap(disparityMap.size(), CV_32FC1);

	const size_t total = disparityMap.total();
	const uint16_t* disparityPtr = disparityMap.ptr<uint16_t>();
	float* depthPtr = depthMap.ptr<float>();

	// Convert disparity to depth
	for (size_t i = 0; i < total; i++) {
		const uint16_t disparityValue = disparityPtr[i];
		const float normalized = disparityValue / maxEncodedValue;

		if (normalized <= 0.f) {
			depthPtr[i] = 0.f;
			continue;
		}

		if (far >= FAR_PLANE_LIMIT_m)
			depthPtr[i] = near / normalized;
		else
			depthPtr[i] = (far * near) / (near + normalized * (far - near));
	}

	return depthMap;
}

cv::Mat PipelineDataLoader::loadDepthMapFromEXR(const std::string& path)
{
	cv::Mat rawDepth = cv::imread(path, cv::IMREAD_UNCHANGED);

	if (rawDepth.empty()) {
		std::cerr << "Error: Could not load depth map from " << path << std::endl;
		return cv::Mat();
	}

	cv::Mat depth;
	// Load first channel only
	if (rawDepth.channels() == 1) {
		depth = rawDepth;
	}
	else {
		cv::extractChannel(rawDepth, depth, 0);
	}

	// Convert if necessary
	if (depth.type() != CV_32F) {
		depth.convertTo(depth, CV_32F);
	}

	return depth;
}

cv::Mat PipelineDataLoader::loadDisparityMapFromYUV(const std::string& yuvDisparityMapPath, const int width, const int height, const int nBitsEncoded) {
	// Validate encoding
	if (nBitsEncoded <= 0 || nBitsEncoded > 16) {
		std::cerr << "Unsupported N_BITS_ENCODED: " << nBitsEncoded << std::endl;
		return cv::Mat();
	}

	// Calculate Y plane size
	const int bytesPerSample = (nBitsEncoded <= 8) ? 1 : 2;
	const size_t yPlaneSizeBytes = static_cast<size_t>(width) * height * bytesPerSample;

	// Pre-allocate Mat to avoid extra allocation and clone
	cv::Mat disparityMap(height, width, CV_16UC1);

	// Open file
	std::ifstream file(yuvDisparityMapPath, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open YUV file: " << yuvDisparityMapPath << std::endl;
		return cv::Mat();
	}

	if (bytesPerSample == 1) {
		// Read 8-bit data
		std::vector<uint8_t> buffer(width * height);
		file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
		if (!file) {
			std::cerr << "Error reading Y plane (8-bit) from " << yuvDisparityMapPath << std::endl;
			return cv::Mat();
		}

		// Expand to 16-bit
		uint16_t* dst = disparityMap.ptr<uint16_t>();
		for (size_t i = 0; i < buffer.size(); ++i)
			dst[i] = static_cast<uint16_t>(buffer[i]) << 8; // fill high bits
	}
	else {
		// Read 16-bit data directly
		file.read(reinterpret_cast<char*>(disparityMap.data), yPlaneSizeBytes);
		if (!file) {
			std::cerr << "Error reading Y plane (16-bit) from " << yuvDisparityMapPath << std::endl;
			return cv::Mat();
		}
	}

	return disparityMap;
}

cv::Mat PipelineDataLoader::loadDepthMap(const std::string& depthMapPath, const DatasetParameters& dataset)
{
	switch (dataset.depthEncoding) {
	case DepthEncoding::FLOAT_METRIC:
		return loadDepthMapFromEXR(depthMapPath);

	case DepthEncoding::MPEG_DISPARITY: {
		// YUV contains MPEG Disparity map that needs conversion
		cv::Mat disparityMap = loadDisparityMapFromYUV(depthMapPath, dataset.depthWidth, dataset.depthHeight, dataset.nBitsEncoded);
		return convertMPEGDisparityToFloatMetric(disparityMap, dataset.nBitsEncoded, dataset.nearPlane_m, dataset.farPlane_m);
	}
	default:
		std::cerr << "Unsupported depth encoding" << std::endl;
		return cv::Mat();
	}
}

cv::Mat PipelineDataLoader::loadRGBImageFromFile(const std::string& rgbPath) {
	cv::Mat rgb = cv::imread(rgbPath, cv::IMREAD_COLOR_RGB);
	if (rgb.empty()) {
		std::cerr << "Error: Could not load RGB image from " << rgbPath << std::endl;
		return cv::Mat();
	}

	// Check channels
	if (rgb.channels() != 3) {
		std::cerr << "Error: RGB image does not have 3 channels from " << rgbPath << std::endl;
		return cv::Mat();
	}
	return rgb;
}

cv::Mat PipelineDataLoader::loadRGBImage(const std::string& rgbPath) {
	// Find RGB image extension
	std::string extension = rgbPath.substr(rgbPath.find_last_of("."));

	// Convert to lowercase for case-insensitive comparison
	std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

	// Execute the right loader
	if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
		extension == ".bmp" || extension == ".tiff") {
		return loadRGBImageFromFile(rgbPath);
	}
	else if (extension == ".yuv") {
		std::cerr << "Error: YUV decoder is not implemented yet." << std::endl;
		return cv::Mat();
	}
	else {
		std::cerr << "Error: Unsupported RGB image format: " << extension << std::endl;
		return cv::Mat();
	}
}
