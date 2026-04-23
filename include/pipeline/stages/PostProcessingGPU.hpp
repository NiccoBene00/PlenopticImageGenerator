#pragma once

#include "pipeline/PipelineStage.hpp"

class PostProcessingGPU : public PipelineStage {
private:
    bool crackFilteringGPU(PipelineData& data);
    bool rotateMicroimagesGPU(PipelineData& data);

protected:
    bool setupSteps() override;

public:
    std::string getStageName() const override {
        return "Post Processing GPU";
    }
};


