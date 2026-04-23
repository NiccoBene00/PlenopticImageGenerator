#pragma once

#include "pipeline/PipelineStage.hpp"

class PointCloudGenerationGPU : public PipelineStage {
private:
    bool initPointCloudGPU(PipelineData& data);
    bool project2Dto3DGPU(PipelineData& data);
    bool adjustPointCloudToSystemGPU(PipelineData& data);

protected:
    bool setupSteps() override;

public:
    std::string getStageName() const override {
        return "Point Cloud Generation GPU";
    }
};