#pragma once
#include "pipeline/PipelineStage.hpp"
#include "data/PipelineData.hpp"
#include <vector>
#include <functional>
#include <string>
#include <iostream>

class PreProcessingGPU : public PipelineStage {
public:
    PreProcessingGPU() = default;
    ~PreProcessingGPU() override = default;

    // Implement the PipelineStage interface
    bool setupSteps() override;
    std::string getStageName() const override {
        return "PreProcessing GPU";
    }

    // Stage functionality
    bool superResolutionGPU(PipelineData& data);

private:
    void setError(const std::string& msg) {
        std::cerr << "[PreProcessingGPU] ERROR: " << msg << std::endl;
    }

    std::vector<std::function<bool(PipelineData&)>> steps;
};