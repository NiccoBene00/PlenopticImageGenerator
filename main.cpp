
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <typeinfo>

#include "pipeline/Pipeline.hpp"
#include "pipeline/PipelineGPU.hpp"
#include "pipeline/PipelineMultiViewGPU.hpp"

// OpenCV logging
#include <opencv2/core/utils/logger.hpp>

#include "data/CameraCalibration.hpp"
#include "data/PipelineData.hpp"
#include "data/MultiViewDatasetLoader.hpp"

//----------------- Program options -----------------
struct ProgramOptions
{
    std::string systemSpecificationPath;
    std::string datasetParametersPath;
    std::string configurationPath;
    std::string outputPath;
    bool guiEnable = false;
};

// Helper per path assoluti
inline std::string getAbsolutePath(std::string relativePath) {
    std::filesystem::path root = PROJECT_ROOT_DIR;
    return (root / relativePath).string();
}

// Parse CLI
static ProgramOptions parseArguments(int argc, char* argv[])
{
    if (argc < 9) {
        throw std::invalid_argument(
            "Usage: " + std::string(argv[0]) +
            " --dataset <file or folder> --system_spec <file> --config <file> --output <folder> [--gui]"
        );
    }

    ProgramOptions options;
    std::unordered_map<std::string, std::string> argMap;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.starts_with("--")) {
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                argMap[key] = argv[i + 1];
                ++i;
            } else {
                argMap[key] = "";
            }
        }
    }

    if (argMap.count("--system_spec") == 0 || argMap.count("--dataset") == 0 ||
        argMap.count("--config") == 0 || argMap.count("--output") == 0) {
        throw std::invalid_argument("Missing required argument.");
    }

    options.systemSpecificationPath = getAbsolutePath(argMap["--system_spec"]);
    options.datasetParametersPath   = getAbsolutePath(argMap["--dataset"]);
    options.configurationPath      = getAbsolutePath(argMap["--config"]);
    options.outputPath             = getAbsolutePath(argMap["--output"]);

    return options;
}

//----------------- File loading helper -----------------
template <class Type>
static Type loadFromFile(const std::string& filepath)
{
    auto data = fromFile<Type>(filepath);
    if (!data) {
        std::cerr << "Failed loading from file: " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }
    return data.value();
}

//----------------- Main -----------------
int main(int argc, char* argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_FATAL);

    ProgramOptions options;
    try {
        options = parseArguments(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    SystemSpec spec = loadFromFile<SystemSpec>(options.systemSpecificationPath);
    Config config = loadFromFile<Config>(options.configurationPath);

    PipelineData data;
    data.spec = spec;
    data.config = config;
    data.outputPath = options.outputPath;

    std::unique_ptr<Pipeline> pipeline;


    std::filesystem::path datasetPath(options.datasetParametersPath);
    if (std::filesystem::is_regular_file(datasetPath)) {
        // ---------- SINGLE VIEW ----------
        std::cout << "[SINGLE VIEW DATASET]\n";
        DatasetParameters dataset = loadFromFile<DatasetParameters>(options.datasetParametersPath);
        data.dataset = dataset;

        if (config.entirePipelineGPU) {
            std::cout << "[GPU SINGLE-VIEW PIPELINE]\n";
            pipeline = std::make_unique<PipelineGPU>(spec, dataset, config, options.outputPath);
        } else {
            std::cout << "[CPU PIPELINE]\n";
            pipeline = std::make_unique<Pipeline>(spec, dataset, config, options.outputPath);
        }
    } else if (std::filesystem::is_directory(datasetPath)) {
        // ---------- MULTI VIEW ----------
		std::cout << "\n";
        std::cout << "[MULTI-VIEW DATASET]";

        if (!DataLoader::loadMultiViewDataset(datasetPath.string(), data)) {
            std::cerr << "[MultiView] Failed to load multi-view dataset\n";
            return EXIT_FAILURE;
        }

		std::cout << "\n";
        std::cout << "[GPU MULTI-VIEW PIPELINE]\n";
        pipeline = std::make_unique<PipelineMultiViewGPU>(spec, data.dataset, config, options.outputPath);
		pipeline->setPipelineData(data);
    } else {
        std::cerr << "Dataset path does not exist: " << datasetPath.string() << "\n";
        return EXIT_FAILURE;
    }


	if (!pipeline->setup()) {
		std::cerr << "[ERROR] Pipeline setup failed\n";
		return EXIT_FAILURE;
	}

    pipeline->initialize();

    // ---------------- Run ----------------
    pipeline->printAllDataSummaries();
    pipeline->runAndSave();
    pipeline->printBenchmarkSummary();

    return EXIT_SUCCESS;
}