
#pragma once

struct PointCloud;
struct DatasetParameters;
struct Config;
struct SystemSpec;

namespace GPU {
namespace PointCloudGPU {

bool project2Dto3D(
    PointCloud& ptCloud,
    const DatasetParameters& dataset,
    const Config& config
);

bool adjustToSystem(
    PointCloud& ptCloud,
    const SystemSpec& spec,
    const Config& config
);

}
}
