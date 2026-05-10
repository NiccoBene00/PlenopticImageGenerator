#pragma once

#include "pipeline/PipelineGPU.hpp"

// Pipeline dedicated to multi-view
class PipelineMultiViewGPU : public PipelineGPU {
public:
    PipelineMultiViewGPU(const SystemSpec& spec,
                         const DatasetParameters& dataset,
                         const Config& config,
                         const std::string& outputPath);

    void initialize() override;

    void setPipelineData(const PipelineData& pipelineData) { this->data = pipelineData; }
protected:
    bool setup() override;
    // Override: yield the specific stage for multi-view 
    void createDefaultStages() override;

    
};