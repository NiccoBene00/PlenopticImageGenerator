/*
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>

#include "pipeline/Pipeline.hpp"
#include "Pipeline/PipelineGPU.hpp"

// REMOVING OPENCV DEBUGGING INFO THAT POLUTES OUTPUT
#include <opencv2/core/utils/logger.hpp>

struct ProgramOptions
{
	std::string systemSpecificationPath; // path to system specification file
	std::string datasetParametersPath;	 // path to file or folder
	std::string configurationPath;		 // path to file or folder
	std::string outputPath;
	bool guiEnable = false;
};

inline std::string getAbsolutePath(std::string relativePath) {
	std::filesystem::path root = PROJECT_ROOT_DIR;
	return (root / relativePath).string();
}

static ProgramOptions parseArguments(int argc, char *argv[])
{
	if (argc < 9 && argc > 10)
	{
		throw std::invalid_argument(
			"Usage: " + std::string(argv[0]) +
			"--gui " +
			"--dataset <parameters json file>.json " +
			"--system_spec <system specification file>.json " +
			"--config <configuration json file>.json"
			"--output <output folder path>");
	}

	ProgramOptions options;
	std::filesystem::path root = PROJECT_ROOT_DIR;

	// Simple map for easier parsing
	std::unordered_map<std::string, std::string> argMap;
	for (int i = 1; i < argc; ++i)
	{
		std::string key = argv[i];
		if (!key.starts_with("--"))
		{
			throw std::invalid_argument("Unexpected argument: " + key);
		}

		// Check if next argument exists and is not another key
		if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
		{

			argMap[key] = argv[i + 1];
			++i; // skip the value
		}
		else
		{
			// Flag without value
			argMap[key] = "";
		}
	}
	

	// Process known keys
	if (argMap.count("--system_spec") == 0)
	{
		throw std::invalid_argument("Missing required argument: --system_spec");
	}
	options.systemSpecificationPath = getAbsolutePath(argMap["--system_spec"]);
	validateJsonExtension(options.systemSpecificationPath, "System specification file");

	if (argMap.count("--dataset") == 0)
	{
		throw std::invalid_argument("Missing required argument: --dataset");
	}
	options.datasetParametersPath = getAbsolutePath(argMap["--dataset"]);
	validateJsonExtension(options.datasetParametersPath, "Dataset parameters file");

	if (argMap.count("--config") == 0)
	{
		throw std::invalid_argument("Missing required argument: --cofig");
	}
	options.configurationPath = getAbsolutePath(argMap["--config"]);
	validateJsonExtension(options.configurationPath, "Configuration file");

	if (argMap.count("--output") == 0)
	{
		throw std::invalid_argument("Missing required argument: --output");
	}
	options.outputPath = getAbsolutePath(argMap["--output"]);

	return options;
}

template <class Type>
static Type loadFromFile(const std::string &filepath)
{
	auto data = fromFile<Type>(filepath);
	if (!data)
	{
		std::cerr << "Failed loading from file: " << filepath << std::endl;
	}
	return data.value();
}

int main(int argc, char *argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_FATAL); // REMOVING OPENCV DEBUGGING INFO THAT POLUTES OUTPUT

	// Parse command line arguments
	ProgramOptions options;
	try
	{
		options = parseArguments(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	SystemSpec spec = loadFromFile<SystemSpec>(options.systemSpecificationPath);
	DatasetParameters data = loadFromFile<DatasetParameters>(options.datasetParametersPath);
	Config config = loadFromFile<Config>(options.configurationPath);



	//-----new-----//
	std::unique_ptr<Pipeline> pipeline;
	if (config.entirePipelineGPU) {
		std::cout << "[GPU PIPELINE]\n";
    	pipeline = std::make_unique<PipelineGPU>(spec, data, config, options.outputPath);
		pipeline->initialize();
	} else {
		std::cout << "[CPU PIPELINE]\n";
    	pipeline = std::make_unique<Pipeline>(spec, data, config, options.outputPath);
		pipeline->initialize();
	}

	pipeline->printAllDataSummaries();
	pipeline->runAndSave();
	pipeline->printBenchmarkSummary();
	//-------------//
	return EXIT_SUCCESS;
}


*/

#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

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
        std::cout << "[MULTI-VIEW DATASET]\n";

        if (!DataLoader::loadMultiViewDataset(datasetPath.string(), data)) {
            std::cerr << "[MultiView] Failed to load multi-view dataset.\n";
            return EXIT_FAILURE;
        }

        std::cout << "[GPU MULTI-VIEW PIPELINE]\n";
        pipeline = std::make_unique<PipelineMultiViewGPU>(spec, data.dataset, config, options.outputPath);
    } else {
        std::cerr << "Dataset path does not exist: " << datasetPath.string() << "\n";
        return EXIT_FAILURE;
    }

    pipeline->initialize();

    // ---------------- Run ----------------
    pipeline->printAllDataSummaries();
    pipeline->runAndSave();
    pipeline->printBenchmarkSummary();

    return EXIT_SUCCESS;
}