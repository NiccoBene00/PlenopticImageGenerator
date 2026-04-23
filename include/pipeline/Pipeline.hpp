#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <string>

#include "PipelineDataLoader.hpp"
#include "PipelineBenchmark.hpp"
#include "data/PipelineData.hpp"

class PipelineStage;

enum class PipelineStatus {
	IDLE,
	RUNNING,
	COMPLETED,
	FAILED
};

class Pipeline {
	protected:
		std::vector<std::unique_ptr<PipelineStage>> stages;
		PipelineStatus status;
		std::string lastErrorMessage;

		PipelineData data;
		PipelineBenchmark benchmark;


		void fail(const std::string& error);
		virtual void createDefaultStages();
		bool loadInputImages();
		void initializePlenopticImage();

public:
	Pipeline(const SystemSpec& spec, const DatasetParameters& dataset, const Config& config, const std::string& outputPath);

	virtual ~Pipeline();

	virtual void initialize();

	PipelineStatus getStatus() const { return status; }
	std::string getLastError() const { return lastErrorMessage; }
	cv::Mat getPlenopticImage() const { if (status == PipelineStatus::COMPLETED && !data.plenopticImage.empty()) return data.plenopticImage; }
	std::vector<std::unique_ptr<PipelineStage>>& getStages() { return stages; }
	double getPipelineDuration() const { return benchmark.getTotalDuration(); }

	bool run();
	bool save() const;
	bool runAndSave();
	bool updateParameters(const SystemSpec& spec, const DatasetParameters& dataset, const Config& config, const std::string& outputPath);

	void reset();

	void printBenchmarkSummary() const;
	void printSystemSpecificationSummary() const;
	void printDatasetParametersSummary() const;
	void printConfigSummary() const;
	void printAllDataSummaries() const;
};