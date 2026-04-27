#pragma once
#include "pipeline/PipelineStage.hpp"
#include "data/PipelineData.hpp"

class MultiViewRegistration : public PipelineStage {
private:
    bool registerPointClouds(PipelineData& data);
    bool mergePointCloudsGPU(PipelineData& data);

protected:
    bool setupSteps() override;

public:
    std::string getStageName() const override {
        return "MultiView Registration GPU";
    }
};