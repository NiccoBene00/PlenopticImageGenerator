#include "pipeline/stages/PointCloudGenerationGPU.hpp"
#include "pipeline/stages/PointCloudGeneration.hpp"
#include "gpu/PointCloudGenerationGPU.cuh"
#include "data/PointCloud.hpp"
#include "data/SystemSpec.hpp"
#include "data/Config.hpp"
#include "data/DatasetParameters.hpp"
#include <iostream>

bool PointCloudGenerationGPU::initPointCloudGPU(PipelineData& data) {
    std::cout << "[GPU] Init Point Cloud\n";

    PointCloudGeneration cpu;
    return cpu.initPointCloud(
        data.inputRgbImage,
        data.inputDepthMap,
        data.pointCloud
    );
}

bool PointCloudGenerationGPU::project2Dto3DGPU(PipelineData& data) {
    std::cout << "[GPU] Project 2D to 3D\n";
    //generatePointCloudCPU(data.pointCloud, data.inputRgbImage, data.inputDepthMap);
    return GPU::PointCloudGPU::project2Dto3D(data.pointCloud, data.dataset, data.config);
}

bool PointCloudGenerationGPU::adjustPointCloudToSystemGPU(PipelineData& data) {
    std::cout << "[GPU] Adjust Point Cloud to System\n";
    return GPU::PointCloudGPU::adjustToSystem(data.pointCloud, data.spec, data.config);
}

bool PointCloudGenerationGPU::setupSteps() {
    steps.clear();

    registerStep(
        "Initialize Point Cloud GPU",
        [this](PipelineData& d) {
            return initPointCloudGPU(d);
        },
        true
    );

    registerStep(
        "Project 2D to 3D GPU",
        [this](PipelineData& d) {
            return project2Dto3DGPU(d);
        },
        true
    );

    registerStep(
        "Adjust Point Cloud to System GPU",
        [this](PipelineData& d) {
            return adjustPointCloudToSystemGPU(d);
        },
        true
    );

    return true;
}