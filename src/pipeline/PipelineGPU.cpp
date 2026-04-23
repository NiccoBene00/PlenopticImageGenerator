#include "pipeline/stages/PreProcessing.hpp"
#include "pipeline/stages/PointCloudGeneration.hpp"
#include "pipeline/stages/PlenopticRendering.hpp"
#include "pipeline/stages/PostProcessingGPU.hpp"
#include "pipeline/stages/PointCloudGenerationGPU.hpp"
#include "pipeline/stages/PreProcessingGPU.hpp"
#include "pipeline/PipelineGPU.hpp"

PipelineGPU::PipelineGPU(const SystemSpec& spec, const DatasetParameters& dataset, const Config& config, 
                        const std::string& outputPath)
    : Pipeline(spec, dataset, config, outputPath)
{
    //createDefaultStages(); 
}

void PipelineGPU::createDefaultStages() {
    stages.clear();

    stages.emplace_back(std::make_unique<PreProcessingGPU>());
    
    //we replace the point cloud generation stage with the GPU version
    stages.emplace_back(std::make_unique<PointCloudGenerationGPU>());

    stages.emplace_back(std::make_unique<PlenopticRendering>());

    //we replace the post processing stage with the GPU version
    stages.emplace_back(std::make_unique<PostProcessingGPU>());
}

void PipelineGPU::initialize() {
    createDefaultStages();
}