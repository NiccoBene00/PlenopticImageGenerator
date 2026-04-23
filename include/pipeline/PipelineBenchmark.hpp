#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <iostream>

class PipelineBenchmark {
	using Clock = std::chrono::high_resolution_clock;

	struct StageInfo {
		std::string getName;
		std::chrono::duration<double, std::milli> duration;
	};

	std::vector<StageInfo> stages;
	Clock::time_point stageStartTime;
public:
	PipelineBenchmark() = default;

	void startStage(const std::string& stageName);
	void endStage();
	void reset();

	double getStageDuration(const std::string& getName) const;
	double getTotalDuration() const;

	void printSummary() const;
};
