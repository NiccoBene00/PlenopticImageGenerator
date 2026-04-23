#pragma once

#include <vector>
#include <functional>
#include <string>
#include "data/PipelineData.hpp"


struct StageStep {
	std::function<bool(PipelineData&)> execute;
	std::string name;
	bool mandatory = false;
	bool enabled = true;

};

class PipelineStage {
protected:
	std::vector<StageStep> steps;
	std::string error;

	void setError(std::string errorMessage) {
		error = std::move(errorMessage);
	}
	void clearError() { error.clear(); }

	void registerStep(const std::string& stepName, 
		std::function<bool(PipelineData&)> fn,
		bool mandatory = false,
		bool enabled = true)
	{
		steps.emplace_back(std::move(fn), stepName, mandatory, enabled);
	}

	virtual bool setupSteps() = 0;

public:
	virtual ~PipelineStage() = default;

	const std::string& getError() const { return error; }
	std::vector<StageStep>& getSteps() { return steps; }

	virtual bool process(PipelineData& data) {
		if (steps.empty()) {
			if (!setupSteps()) {
				setError("Failed to setup steps of stage");
				return false;
			}
		}

		clearError();
		for (auto& step : steps) {
			if (!step.mandatory && !step.enabled) continue;

			if (!step.execute(data)) {
				setError("Step Failed: " + step.name);
				return false;
			}
		}
		return true;
	}

	virtual std::string getStageName() const = 0;
};