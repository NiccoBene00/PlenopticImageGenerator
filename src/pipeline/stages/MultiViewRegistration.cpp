#include "pipeline/stages/MultiViewRegistration.hpp"
#include "gpu/MultiViewRegistration.cuh"
#include <iostream>

bool MultiViewRegistration::registerPointClouds(PipelineData& data) {
    std::cout << "[GPU] Registering point clouds...\n";
    return GPU::MultiViewRegistration::loadAndTransformPointClouds(data);
}

bool MultiViewRegistration::mergePointCloudsGPU(PipelineData& data) {
    std::cout << "[GPU] Merging and deduplicating point clouds...\n";
    return GPU::MultiViewRegistration::mergeAndDeduplicate(data);
}

bool MultiViewRegistration::setupSteps() {
    steps.clear();

    registerStep(
        "Register Multi-View Point Clouds",
        [this](PipelineData& data){ return registerPointClouds(data); },
        true
    );

    registerStep(
        "Merge & Deduplicate Point Clouds",
        [this](PipelineData& data){ return mergePointCloudsGPU(data); },
        true
    );

    return true;
}