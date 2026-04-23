#pragma once

#define EIGEN_NO_DEBUG

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

typedef Eigen::Array<float, Eigen::Dynamic, 1> ArrayF;
typedef Eigen::Array<uint16_t, Eigen::Dynamic, 1> ArrayU16;
typedef Eigen::Array<uint8_t, Eigen::Dynamic, 1> ArrayU8;
typedef Eigen::Array<cv::Vec3b, Eigen::Dynamic, 1> ArrayRGB;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> Mask;

inline ArrayU8 extractChannel(const ArrayRGB& arr, int channel)
{
	assert(channel >= 0 && channel < 3);
	if (arr.size() == 0) return ArrayU8();
	ArrayU8 out(arr.size());
	for (Eigen::Index i = 0; i < arr.size(); ++i)
		out(i) = arr(i)[channel];
	return out;
}

template <typename Derived>
inline float getSizeEigen(const Eigen::ArrayBase<Derived>& arr) {
	if (arr.size() == 0) return 0.0f;
	const auto evalArr = arr.eval();
	return evalArr.maxCoeff() - evalArr.minCoeff();
}

template <typename Derived>
inline float getCenterEigen(const Eigen::ArrayBase<Derived>& arr) {
	if (arr.size() == 0) return 0.0f;
	const auto evalArr = arr.eval();
	return (evalArr.maxCoeff() + evalArr.minCoeff()) / 2.0f;
}

template <typename Derived>
inline float getMeanEigen(const Eigen::ArrayBase<Derived>& arr) {
	if (arr.size() == 0) return 0.0f;
	const auto evalArr = arr.eval();
	return evalArr.mean();
}

template <typename Derived>
inline float getMinEigen(const Eigen::ArrayBase<Derived>& arr) {
	if (arr.size() == 0) return 0.0f;
	return arr.minCoeff();
}

template <typename Derived>
inline float getMaxEigen(const Eigen::ArrayBase<Derived>& arr) {
	if (arr.size() == 0) return 0.0f;
	return arr.maxCoeff();
}

template<typename Derived, typename MaskType>
Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1>
maskSelect(const Eigen::ArrayBase<Derived>& arr, const MaskType& mask)
{
	// This is pretty useful for one single mask 
	// For masking several vectors, implement it by hand
	assert(arr.size() == mask.size());
	Eigen::Index count = mask.count();
	Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1> out(count);
	Eigen::Index j = 0;
	for (Eigen::Index i = 0; i < arr.size(); ++i)
		if (mask(i)) out(j++) = arr(i);
	return out;
}
