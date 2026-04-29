#include "pipeline/Pipeline.hpp"

#include <iostream>
#include <iomanip>

#include "pipeline/stages/PreProcessing.hpp"
#include "pipeline/stages/PointCloudGeneration.hpp"	
#include "pipeline/stages/PlenopticRendering.hpp"
#include "pipeline/stages/PostProcessing.hpp"
#include "pipeline/PipelineStage.hpp"

#include "utils/PrintUtils.hpp"
#include "utils/OutputColors.hpp"

Pipeline::~Pipeline() = default;

Pipeline::Pipeline(const SystemSpec& spec, const DatasetParameters& dataset, const Config& config, const std::string& outputPath)
	: status(PipelineStatus::IDLE)
{
	data.spec = spec;
	data.dataset = dataset;
	data.config = config;
	data.outputPath = outputPath;

	/*
	if (!loadInputImages()) {
		fail("Pipeline initialization failed");
		return;
	}

	initializePlenopticImage();
	//createDefaultStages();
	*/
}

void Pipeline::setPipelineData(const PipelineData& data){

}

bool Pipeline::setup()
{
    if (!loadInputImages()) {
        fail("Pipeline initialization failed");
        return false;
    }

    initializePlenopticImage();
    return true;
}

void Pipeline::initialize() {
    if (status == PipelineStatus::FAILED) {
        return;
    }

    createDefaultStages();
}

void Pipeline::createDefaultStages() {
	stages.emplace_back(std::make_unique<PreProcessing>());
	stages.emplace_back(std::make_unique<PointCloudGeneration>());
	stages.emplace_back(std::make_unique<PlenopticRendering>());
	stages.emplace_back(std::make_unique<PostProcessing>());
}

void Pipeline::fail(const std::string& error) {
	status = PipelineStatus::FAILED;
	lastErrorMessage = error;
	std::cerr << color::BOLD_RED << "Error: " << lastErrorMessage << color::RESET << std::endl;
}

void Pipeline::reset() {
	status = PipelineStatus::IDLE;
	initializePlenopticImage();
	data.pointCloud.clear();
	lastErrorMessage.clear();
	benchmark.reset();
}

bool Pipeline::run() {
	if (stages.empty()) {
		fail("No pipeline stages configured.");
		return false;
	}

	if (status == PipelineStatus::FAILED) {
		fail("Pipeline is unable to run due to previous failures.");
		return false;
	}

	// Clear previous pipeline data
	reset();

	// Run each stage in sequence
	int idx = 0;
	status = PipelineStatus::RUNNING;
	std::string stageName;
	for (auto& stage : stages) {
		stageName = stage->getStageName();

		printTitle("Running Stage " + std::to_string(idx++) + ": " + stageName, PRINT_WIDTH);
		benchmark.startStage(stageName);
		if (!stage->process(data)) {
			benchmark.endStage();

			fail("[" + stageName + "]: " + stage->getError());
			return false;
		}
		benchmark.endStage();
	}

	status = PipelineStatus::COMPLETED;
	return true;
}

bool Pipeline::save() const
{
	if (status != PipelineStatus::COMPLETED) {
		std::cerr << color::BOLD_RED
			<< "Error: Cannot save - pipeline has not completed successfully"
			<< color::RESET << std::endl;
		return false;
	}

	cv::Mat outputRGB;
	cv::cvtColor(data.plenopticImage, outputRGB, cv::COLOR_BGRA2RGB);

	if (!cv::imwrite(data.outputPath, outputRGB)) {
		std::cerr << color::BOLD_RED
			<< "Error: Couldn't save the plenoptic image to: "
			<< data.outputPath << color::RESET << std::endl;
		return false;
	}

	std::cout << color::BOLD_GREEN << "Plenoptic Image saved to: " << data.outputPath << color::RESET << std::endl;
	return true;
}

bool Pipeline::runAndSave() {
	if (!run()) return false;

	if (!save()) return false;

	return true;
}

bool Pipeline::loadInputImages() {
	PipelineDataLoader loader;

	// Load and validate images
	data.inputRgbImage = loader.loadRGBImage(data.dataset.rgbImagePath);
	if (data.inputRgbImage.empty()) {
		fail("Could not load input image from " + data.dataset.rgbImagePath);
		return false;
	}


	data.inputDepthMap = loader.loadDepthMap(data.dataset.depthMapPath, data.dataset);
	if (data.inputDepthMap.empty()) {
		fail("Could not load depth map from " + data.dataset.depthMapPath);
		return false;
	}

	if (data.inputRgbImage.size() != data.inputDepthMap.size()) {
		fail("Input image and depth map sizes do not match.");
		return false;
	}

	return true;
}

void Pipeline::initializePlenopticImage()
{
	data.plenopticImage = cv::Mat(data.spec.display.resolutionY, data.spec.display.resolutionX, CV_8UC4, cv::Scalar(0, 0, 0, 0));
}

bool Pipeline::updateParameters(const SystemSpec& spec, const DatasetParameters& dataset, const Config& config, const std::string& outputPath) {
	if (status == PipelineStatus::RUNNING) {
		fail("Cannot update parameters while pipeline is running");
		return false;
	}

	// Update pipeline data
	data.spec = spec;
	data.dataset = dataset;
	data.config = config;
	data.outputPath = outputPath;

	reset();
	return true;
}

void Pipeline::printBenchmarkSummary() const {
	benchmark.printSummary();
}

void Pipeline::printSystemSpecificationSummary() const {
	std::vector<std::pair<std::string, std::string>> rows = {
		{"System Name", data.spec.name},
		{"Display Resolution", pairToString(data.spec.display.resolutionX, data.spec.display.resolutionY)},
		{"Display Size (mm)", pairToString(data.spec.display.width_mm, data.spec.display.height_mm)},
		{"MLA Type", data.spec.mla.type},
		{"MLA Pitch (mm)", std::to_string(data.spec.mla.pitch_mm)},
		{"MLA Focal Length (mm)", std::to_string(data.spec.mla.focalLength_mm)},
		{"MLA Distance (mm)", std::to_string(data.spec.mla.displayDistance_mm)},
		{"MLA Count", pairToString(data.spec.mla.countX, data.spec.mla.countY)}
	};
	printTable("System Specification Summary", rows, PrintAlign::LL, PRINT_WIDTH);
}

void Pipeline::printDatasetParametersSummary() const {
	std::vector<std::pair<std::string, std::string>> rows = {
		{"Dataset Name", data.dataset.datasetName},
		{"Input Image Path", data.dataset.rgbImagePath},
		{"Depth Map Path", data.dataset.depthMapPath},
		{"Depth Encoding", DepthEncodingParser.toString(data.dataset.depthEncoding)},
		{"Camera Focal Length [px]", pairToString(data.dataset.CAM_FX_px, data.dataset.CAM_FY_px)},
		{"Camera Principal Point [px]", pairToString(data.dataset.CAM_PX_px, data.dataset.CAM_PY_px)},
		{"Input Image Size [px]", makeImageInfo(data.inputRgbImage)},
		{"Depth Map Size [px]", makeImageInfo(data.inputDepthMap)},
		{"Near Plane [mm]", std::to_string(data.dataset.nearPlane_m)},
		{"Far Plane [mm]", std::to_string(data.dataset.farPlane_m)},
		{"Num Bit Encoded", std::to_string(data.dataset.nBitsEncoded)}
	};

	printTable("Pipeline Data Summary", rows, PrintAlign::LL, PRINT_WIDTH);
}

void Pipeline::printConfigSummary() const {
	printTitle("Pipeline Config Summary", PRINT_WIDTH);
	printLabelValue("Super Resolution Scale", data.config.superResolutionFactor);
	printLabelValue("Display Mode", PointCloudModeParser.toString(data.config.pointCloudMode));
	printLabelValue("Entire Pipeline on GPU (1=yes, 0=no)", data.config.entirePipelineGPU);
	printEndSeparator(PRINT_WIDTH);
}

void Pipeline::printAllDataSummaries() const
{
	printSystemSpecificationSummary();
	printDatasetParametersSummary();
	printConfigSummary();
}
