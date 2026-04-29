#pragma once

#include "pipeline/Pipeline.hpp"

class PipelineGPU : public Pipeline {
public:
    PipelineGPU(const SystemSpec& spec,
                const DatasetParameters& dataset,
                const Config& config,
                const std::string& outputPath);


    void initialize() override;

    void setPipelineData(const PipelineData& data) override;

protected:
    void createDefaultStages() override;
};