
#include "gpu/PostProcessingGPU.cuh"
#include "pipeline/stages/PostProcessing.hpp"
#include "pipeline/stages/PostProcessingGPU.hpp"

#include <iostream>

bool PostProcessingGPU::crackFilteringGPU(PipelineData& data)
{
    std::cout << "[GPU] Crack Filtering\n";
    //return GPU::PostProcessing::crackFiltering(data.plenopticImage, data.spec, data.config);
    PostProcessing cpuPost;
    cpuPost.computeMicroimagesRegions(
        data.spec,
        data.plenopticImage.cols,
        data.plenopticImage.rows
    );

    auto microGPU = cpuPost.getMicroimagesGPU();

    return GPU::PostProcessing::crackFiltering(
        data.plenopticImage,
        microGPU,
        data.config
    );
}

bool PostProcessingGPU::rotateMicroimagesGPU(PipelineData& data)
{
    std::cout << "[GPU] Rotation\n";
    int microimageSize = data.spec.display.resolutionX / data.spec.mla.countX;

    std::cout << "ResolutionX: " << data.spec.display.resolutionX << std::endl;
    std::cout << "MLA countX: " << data.spec.mla.countX << std::endl;
    std::cout << "microimageSize: " << microimageSize << std::endl;
    return GPU::PostProcessing::rotateMicroimages(data.plenopticImage, data.spec);
}

bool PostProcessingGPU::setupSteps()
{
    steps.clear();

    registerStep(
        "Crack Filtering GPU",
        [this](PipelineData& d) {
            return crackFilteringGPU(d);
        },
        true
    );

    registerStep(
        "Rotate Microimages GPU",
        [this](PipelineData& d) {
            return rotateMicroimagesGPU(d);
        },
        true
    );

    return true;
}



