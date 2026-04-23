#include "pipeline/PipelineBenchmark.hpp"

#include "utils/PrintUtils.hpp"
#include "utils/OutputColors.hpp"

void PipelineBenchmark::startStage(const std::string& stageName){
	stageStartTime = Clock::now();
	stages.emplace_back(
		StageInfo{
			stageName,
			std::chrono::duration<double, std::milli>(0)
		}
	);
}

void PipelineBenchmark::endStage(){
	Clock::time_point endTime = Clock::now();
	stages.back().duration = endTime - stageStartTime;
}

void PipelineBenchmark::reset(){
	stages.clear();
}

double PipelineBenchmark::getStageDuration(const std::string& getName) const{
	for (const auto& stage : stages) {
		if (stage.getName == getName) {
			return stage.duration.count(); // returns duration in milliseconds
		}
	}

	// Stage not found
	std::cerr << "Warning: Stage '" << getName << "' not found in benchmark.\n";
	return 0.0;

}

double PipelineBenchmark::getTotalDuration() const {
	double total = 0.0;
	for (const auto& s : stages) {
		total += s.duration.count();
	}
	return total;
}

void PipelineBenchmark::printSummary() const {
	std::cout << color::BOLD_BLUE;

	printTitle("Pipeline Benchmark Summary", PRINT_WIDTH);

	const double totalDuration = getTotalDuration();
	printLabelValue("Total Execution Time (ms)", totalDuration);

	// Build stage table
	std::vector<std::pair<std::string, std::string>> rows;

	for (const auto& stage : stages) {
		std::string stageName = stage.getName;

		float percentage = (totalDuration > 0.0)
			? static_cast<float>((stage.duration.count() / totalDuration) * 100.0)
			: 0.0f;

		std::ostringstream percentOss;
		percentOss << std::fixed << std::setprecision(2) << percentage;
		stageName += " (" + percentOss.str() + "%)";

		std::ostringstream durationOss;
		durationOss << std::fixed << std::setprecision(2)
			<< stage.duration.count();

		rows.emplace_back(stageName, durationOss.str());
	}

	printTable("Stage Execution Times (ms)", rows, PrintAlign::LL, PRINT_WIDTH);

	std::cout << color::RESET;
}

