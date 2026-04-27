#pragma once

#include "pipeline/PipelineGPU.hpp"

// Pipeline dedicata al multi-view
class PipelineMultiViewGPU : public PipelineGPU {
public:
    PipelineMultiViewGPU(const SystemSpec& spec,
                         const DatasetParameters& dataset,
                         const Config& config,
                         const std::string& outputPath);

    // Override: crea le stage specifiche per multi-view
    void createDefaultStages() override;
};