#pragma once

#include <iomanip>
#include <fstream>
#include <sstream>

#include "utils/types.hpp"
#include "utils/JSONUtils.hpp"
#include "utils/OutputColors.hpp"

struct PointCloudStats {
	// 3D coordinates
	float xMin, xMax, xCenter, xMean, xRange;
	float yMin, yMax, yCenter, yMean, yRange;
	float zMin, zMax, zCenter, zMean, zRange;
};

struct PointCloud {
	// Convention: Uppercase letters are in mm and lowercase ones are px
	ArrayF X, Y, Z;		// 3D coordinates in mm
	ArrayU16 px, py;	// pixel coordinates in the image
	ArrayRGB colors;

	void clear() { if (size() > 0) resize(0); }

	size_t size() const { return Z.size(); }

	inline cv::Vec3b getColor(size_t idx) const {
		return colors(idx);
	}

	inline std::string sizeString() const {
		std::ostringstream oss;
		oss << "PointCloud("
			<< "X=" << X.size() << ", "
			<< "Y=" << Y.size() << ", "
			<< "Z=" << Z.size() << ", "
			<< "px=" << px.size() << ", "
			<< "py=" << py.size() << ","
			<< "colors=" << colors.size() << ")";
		return oss.str();
	}

	inline void showImage() const {
		cv::Mat img(1080 * 2, 1920 * 2, CV_8UC3, cv::Scalar(0, 0, 0));
		for (size_t i = 0; i < size(); i++)
		{
			img.at<cv::Vec3b>(py[i], px[i]) = colors[i];
		}
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		cv::resize(img, img, cv::Size(1920, 1080));
		cv::imshow("Point Cloud Image", img);
	}

	inline std::string statsString() const {
		std::ostringstream oss;
		oss << "Stats:\n"
			<< "X: min=" << getMinX() << ", max=" << getMaxX() << ", mean=" << getMeanX() << ", center=" << getCenterX() << "\n"
			<< "Y: min=" << getMinY() << ", max=" << getMaxY() << ", mean=" << getMeanY() << ", center=" << getCenterY() << "\n"
			<< "Z: min=" << getMinZ() << ", max=" << getMaxZ() << ", mean=" << getMeanZ() << ", center=" << getCenterZ() << "\n";
		return oss.str();
	}

	inline void resize(size_t size) {
		X.resize(size);
		Y.resize(size);
		Z.resize(size);
		px.resize(size);
		py.resize(size);
		colors.resize(size);
	}

	inline void conservativeResize(size_t size) {
		X.conservativeResize(size);
		Y.conservativeResize(size);
		Z.conservativeResize(size);
		px.conservativeResize(size);
		py.conservativeResize(size);
	}

	// -------- X getters --------
	inline float getMinX()    const { return X.size() ? X.minCoeff() : 0.f; }
	inline float getMaxX()    const { return X.size() ? X.maxCoeff() : 0.f; }
	inline float getMeanX()   const { return X.size() ? X.mean() : 0.f; }
	inline float getCenterX() const { return 0.5f * (getMinX() + getMaxX()); }
	inline float getRangeX()  const { return getMaxX() - getMinX(); }

	// -------- Y getters --------
	inline float getMinY()    const { return Y.size() ? Y.minCoeff() : 0.f; }
	inline float getMaxY()    const { return Y.size() ? Y.maxCoeff() : 0.f; }
	inline float getMeanY()   const { return Y.size() ? Y.mean() : 0.f; }
	inline float getCenterY() const { return 0.5f * (getMinY() + getMaxY()); }
	inline float getRangeY()  const { return getMaxY() - getMinY(); }

	// -------- Z getters --------
	inline float getMinZ()    const { return Z.size() ? Z.minCoeff() : 0.f; }
	inline float getMaxZ()    const { return Z.size() ? Z.maxCoeff() : 0.f; }
	inline float getMeanZ()   const { return Z.size() ? Z.mean() : 0.f; }
	inline float getCenterZ() const { return 0.5f * (getMinZ() + getMaxZ()); }
	inline float getRangeZ()  const { return getMaxZ() - getMinZ(); }

	PointCloudStats computeStats() const {
		PointCloudStats stats;

		// Initialize
		float xMin = X[0], xMax = X[0], yMin = Y[0], yMax = Y[0], zMin = Z[0], zMax = Z[0];
		double sumX = X[0], sumY = Y[0], sumZ = Z[0];

		const float* Xptr = X.data();
		const float* Yptr = Y.data();
		const float* Zptr = Z.data();

		// Single loop over all points, no allocations
		const size_t total = size();
		for (size_t i = 1; i < total; ++i) {
			const float xi = Xptr[i], yi = Yptr[i], zi = Zptr[i];

			xMin = std::min(xMin, xi);
			xMax = std::max(xMax, xi);
			sumX += xi;

			yMin = std::min(yMin, yi);
			yMax = std::max(yMax, yi);
			sumY += yi;

			zMin = std::min(zMin, zi);
			zMax = std::max(zMax, zi);
			sumZ += zi;
		}

		stats.xMin = xMin;  stats.xMax = xMax; stats.xMean = float(sumX / total);
		stats.yMin = yMin;  stats.yMax = yMax; stats.yMean = float(sumY / total);
		stats.zMin = zMin;  stats.zMax = zMax; stats.zMean = float(sumZ / total);

		stats.xCenter = 0.5f * (xMin + xMax);
		stats.yCenter = 0.5f * (yMin + yMax);
		stats.zCenter = 0.5f * (zMin + zMax);

		stats.xRange = xMax - xMin;
		stats.yRange = yMax - yMin;
		stats.zRange = zMax - zMin;

		return stats;
	}



};

static bool savePointCloud(const PointCloud& ptCloud, const std::string& filename) {
	// Resolve path
	fs::path pathToWrite;
	auto resolved = resolvePath(filename);
	if (resolved) {
		pathToWrite = *resolved;
	}
	else {
		fs::path root(PROJECT_ROOT_DIR);
		pathToWrite = root / filename;
	}

	// Ensure directory exists
	fs::create_directories(pathToWrite.parent_path());

	std::ofstream file(pathToWrite, std::ios::out | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open file for writing: " << pathToWrite << std::endl;
		return false;
	}

	const size_t n = ptCloud.size();

	// --- Write PLY header ---
	file << "ply\n"
		<< "format binary_little_endian 1.0\n"
		<< "element vertex " << n << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "property ushort u\n"
		<< "property ushort v\n"
		<< "end_header\n";

	// --- Buffered binary writing ---
	const size_t bufferPoints = 10000; // number of points per buffer
	std::vector<char> buffer;
	buffer.resize(bufferPoints * (3 * sizeof(float) + 2 * sizeof(unsigned short)));

	size_t idx = 0;
	for (size_t i = 0; i < n; ++i) {
		float x = ptCloud.X[i];
		float y = ptCloud.Y[i];
		float z = ptCloud.Z[i];
		unsigned short u = ptCloud.px[i];
		unsigned short v = ptCloud.py[i];

		char* ptr = buffer.data() + idx * (3 * sizeof(float) + 2 * sizeof(unsigned short));
		std::memcpy(ptr, &x, sizeof(float));
		ptr += sizeof(float);
		std::memcpy(ptr, &y, sizeof(float));
		ptr += sizeof(float);
		std::memcpy(ptr, &z, sizeof(float));
		ptr += sizeof(float);
		std::memcpy(ptr, &u, sizeof(unsigned short));
		ptr += sizeof(unsigned short);
		std::memcpy(ptr, &v, sizeof(unsigned short));

		++idx;

		if (idx >= bufferPoints) {
			file.write(buffer.data(), idx * (3 * sizeof(float) + 2 * sizeof(unsigned short)));
			idx = 0;
		}
	}

	// Flush remaining points
	if (idx > 0) {
		file.write(buffer.data(), idx * (3 * sizeof(float) + 2 * sizeof(unsigned short)));
	}

	file.close();
	std::cout << color::bold_green("Binary PLY file successfully saved: " + pathToWrite.string()) << std::endl;
	return true;
}
