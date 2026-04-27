#include "pipeline/PipelineMultiViewGPU.hpp"

#include "pipeline/stages/PreProcessingGPU.hpp"
#include "pipeline/stages/MultiViewPointCloud.hpp"
#include "pipeline/stages/MultiViewRegistration.hpp"
#include "pipeline/stages/PlenopticRendering.hpp"
#include "pipeline/stages/PostProcessingGPU.hpp"
#include "pipeline/stages/PointCloudGenerationGPU.hpp"

PipelineMultiViewGPU::PipelineMultiViewGPU(const SystemSpec& spec,
                                           const DatasetParameters& dataset,
                                           const Config& config,
                                           const std::string& outputPath)
    : PipelineGPU(spec, dataset, config, outputPath)
{
    // Non chiamiamo createDefaultStages qui: lo facciamo separatamente
}

void PipelineMultiViewGPU::createDefaultStages() {
    stages.clear();

    // Pre-processing GPU
    stages.emplace_back(std::make_unique<PreProcessingGPU>());

    // Multi-view point cloud generation
    stages.emplace_back(std::make_unique<MultiViewPointCloud>(data.calibration));

    // Multi-view registration
    stages.emplace_back(std::make_unique<MultiViewRegistration>());

    // Rendering e post-processing
    stages.emplace_back(std::make_unique<PlenopticRendering>());
    stages.emplace_back(std::make_unique<PostProcessingGPU>());
}