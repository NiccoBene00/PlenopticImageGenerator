#include "pipeline/stages/PostProcessing.hpp"
#include "gpu/PostProcessingGPU.cuh"

#include <iostream>
#include <algorithm>
#include <atomic>

bool PostProcessing::computeMicroimagesRegions(const SystemSpec& spec, const int imgWidth, const int imgHeight)
{
	
	microimages.clear();

	const int mlaWidth = spec.mla.countX;
	const int mlaHeight = spec.mla.countY;
	const int totalLenses = spec.mla.totalLenses;

	const int microimageSize_px = static_cast<int>(std::round(spec.mla.pitch_mm / spec.display.pixelSize_mm));

	if (microimageSize_px <= 0) {
		setError("Computed microimage size is invalid: " + std::to_string(microimageSize_px));
		return false;
	}

	microimages.resize(totalLenses);
	int validCount = 0;
	for (int mlaIdx = 0; mlaIdx < totalLenses; mlaIdx++) {
		const int mi = mlaIdx % mlaWidth;
		const int mj = mlaIdx / mlaWidth;

		int start_mcx = mi * microimageSize_px;
		int start_mcy = mj * microimageSize_px;
		int end_mcx = std::min(imgWidth, start_mcx + microimageSize_px);
		int end_mcy = std::min(imgHeight, start_mcy + microimageSize_px);

		int width = end_mcx - start_mcx;
		int height = end_mcy - start_mcy;
		if (width <= 0 || height <= 0) continue;

		microimages[validCount++] = { cv::Rect(start_mcx, start_mcy, width, height), mi, mj };
	}

	// trim unused slots
	microimages.resize(validCount);

	if (microimages.empty()) {
		setError("No valid microimage regions found. Check MLA counts, pitch, and image size.");
		return false;
	}

	std::cout << "Post-Processing initialized with " << microimages.size() << " microimage regions." << std::endl;
	
	return true;
	
}

bool PostProcessing::filterCrackArtifacts(const SystemSpec& spec, const Config& config, cv::Mat& plenopticImage)
{
	
	if (plenopticImage.empty()) {
		setError("Plenoptic image is empty.");
		return false;
	}

	if (config.crackFilteringKernel < 3 || config.crackFilteringKernel % 2 == 0) {
		setError("Crack Filtering Kernel must be odd and >= 3.");
		return false;
	}

	if (microimages.empty()) {
		setError("Crack Filtering: No microimage regions available.");
		return false;
	}

	// Extract alpha channel once for full image
	cv::Mat alpha, crackMask;
	cv::extractChannel(plenopticImage, alpha, 3);
	cv::compare(alpha, 0, crackMask, cv::CMP_EQ);

	int totalCrackPixels = cv::countNonZero(crackMask);
	if (totalCrackPixels == 0) {
		setError("Crack Filtering: No cracks detected in plenoptic image.");
		return false;
	}

	std::atomic<int> crackPixelsProcessed(0);

	// Parallel processing of microimages
	cv::parallel_for_(cv::Range(0, static_cast<int>(microimages.size())),
		[&](const cv::Range& range) {

			// Per-thread Mats to avoid race conditions
			cv::Mat microimage, microMask, interpolation;

			for (int i = range.start; i < range.end; ++i) {
				const auto& region = microimages[i];

				microimage = plenopticImage(region.rect); // avoid reading pixel outside the img
				if (microimage.empty()) continue;

				microMask = crackMask(region.rect);
				int nCracks = cv::countNonZero(microMask);
				if (nCracks == 0) continue;

				crackPixelsProcessed += nCracks;

				// Median filter the microimage
				cv::medianBlur(microimage, interpolation, config.crackFilteringKernel);

				// Copy only the crack pixels back
				interpolation.copyTo(microimage, microMask);
			}
		});

	std::cout << "Crack Filtering:\n\t* Total of crack pixels processed: " << crackPixelsProcessed << std::endl;

	

	return true;
}

bool PostProcessing::rotateMicroimage180(cv::Mat& plenopticImage)
{
	
	
	// THIS FUNCTION MESSES UP SIMULATION. IMPLEMENT IT BUT NOT IMPORTANT FOR NOW. DISCUSS WITH ME LATER.
	if (microimages.empty()) {
		setError("Rotate Microimage 180 deg: No microimage regions available.");
		return false;
	}

#pragma omp parallel for
	for (int idx = 0; idx < static_cast<int>(microimages.size()); ++idx) {
		cv::Mat microimage = plenopticImage(microimages[idx].rect);

		// in-place flip manually
		int rows = microimage.rows;
		int cols = microimage.cols;
		int ch = microimage.channels();

		for (int y = 0; y < rows / 2; ++y) {
			uchar* rowTop = microimage.ptr<uchar>(y);
			uchar* rowBottom = microimage.ptr<uchar>(rows - 1 - y);
			for (int x = 0; x < cols; ++x) {
				for (int c = 0; c < ch; ++c) {
					std::swap(rowTop[x * ch + c], rowBottom[(cols - 1 - x) * ch + c]);
				}
			}
		}

		if (rows % 2) {
			int y = rows / 2;
			uchar* row = microimage.ptr<uchar>(y);
			for (int x = 0; x < cols / 2; ++x) {
				for (int c = 0; c < ch; ++c) {
					std::swap(row[x * ch + c], row[(cols - 1 - x) * ch + c]);
				}
			}
		}
	}

	std::cout << "Rotation 180 deg:\n\t* Microimages rotated: " << microimages.size() << std::endl;
	
	return true;
}

bool PostProcessing::setupSteps()
{
	steps.clear();

	registerStep(
		"Compute Microimage Region",
		[this](PipelineData& d) {
			return computeMicroimagesRegions(d.spec, d.plenopticImage.cols, d.plenopticImage.rows);
		},
		true
	);

	registerStep(
		"Filter Crack Artifacts",
		[this](PipelineData& d) {
			return filterCrackArtifacts(d.spec, d.config, d.plenopticImage);

			
			//-------NEW-------
			auto microGPU = getMicroimagesGPU();

			return GPU::PostProcessing::crackFiltering(
				d.plenopticImage,
				microGPU,
				d.config
			);
			//-----------------
			
			
		});

	registerStep(
		"Rotate Microimage 180 deg (Okano)",
		[this](PipelineData& d) {
			return rotateMicroimage180(d.plenopticImage);
		});

	return true;
}


//--------NEW---------
std::vector<GPU::PostProcessing::MicroimageGPU> PostProcessing::getMicroimagesGPU() const
{
    std::vector<GPU::PostProcessing::MicroimageGPU> out;
    out.reserve(microimages.size());

    for (const auto& m : microimages) {
        GPU::PostProcessing::MicroimageGPU g;
        g.x = m.rect.x;
        g.y = m.rect.y;
        g.width  = m.rect.width;
        g.height = m.rect.height;
        out.push_back(g);
    }

    return out;
}
//--------------------
