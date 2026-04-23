#pragma once
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>


constexpr auto PRINT_WIDTH = 70;
inline constexpr int LABEL_WIDTH = 25;

enum class PrintAlign {
	LL, LR, RL, RR
};

template <typename T, typename U>
inline std::string pairToString(const T& a, const U& b) {
	return "(" + std::to_string(a) + ", " + std::to_string(b) + ")";
}

inline std::string makeImageInfo(const cv::Mat& img) {
	return img.empty() ? "0x0" : std::to_string(img.cols) + "x" + std::to_string(img.rows) + " Channels: " + std::to_string(img.channels());
}

inline std::string cleanTypeName(const std::string& name) {
	if (name.rfind("class ", 0) == 0) return name.substr(6);
	if (name.rfind("struct ", 0) == 0) return name.substr(7);
	return name;
}

inline void printTitle(const std::string& title, size_t width, char sep = '=') {
	size_t padding = (width > title.size() + 2) ? (width - title.size() - 2) / 2 : 0;
	std::cout << std::string(padding, sep) << " " << title << " " << std::string(padding, sep) << "\n";
}

inline void printSeparator(size_t width) { std::cout << std::string(width, '-') << "\n"; }
inline void printEndSeparator(size_t width) { std::cout << std::string(width, '=') << std::endl; }

template <typename T>
inline void printLabelValue(const std::string& label, const T& value) {
	std::cout << std::left << std::setw(LABEL_WIDTH) << label << ": " << value << "\n";
}

template <typename T>
inline void printKeyValue(const std::string& label, const T& value,
	PrintAlign align = PrintAlign::LL,
	size_t labelWidth = LABEL_WIDTH,
	size_t valueWidth = 15)
{
	auto printAligned = [&](auto& valAlign, auto& keyAlign) {
		std::cout << keyAlign << std::setw(labelWidth) << label << " : " << valAlign << std::setw(valueWidth) << value << "\n";
		};

	switch (align) {
	case PrintAlign::LL: printAligned(std::left, std::left); break;
	case PrintAlign::LR: printAligned(std::right, std::left); break;
	case PrintAlign::RL: printAligned(std::left, std::right); break;
	case PrintAlign::RR: printAligned(std::right, std::right); break;
	}
}

inline void printTable(const std::string& title,
	const std::vector<std::pair<std::string, std::string>>& rows,
	PrintAlign align = PrintAlign::LL,
	size_t width = 80)
{
	if (rows.empty()) return;
	size_t keyWidth = 0;
	size_t valueWidth = 0;

	// 70% - 30% split
	keyWidth = static_cast<size_t>(width * 0.7) - 3;
	valueWidth = width - keyWidth - 3;

	if (!title.empty()) printTitle(title, width);

	for (const auto& [key, value] : rows) {
		printKeyValue(key, value, align, keyWidth, valueWidth);
	}

	printEndSeparator(width);
}

