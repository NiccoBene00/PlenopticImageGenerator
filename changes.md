# Optimization changes

## Notes

Phase 1
I create a toggle system that is able to switch pipeline mode (previuous mode where only the rendering stage was on GPU and 
the new one where we aim to bring each stage on GPU).

Phase 2
I analyze how bringing everything on GPU. First I observe that the most expensive stage is the Post Processing one (above 50% of the total stage execution time), so I focus on creating a GPU Post Processing stage.

Phase 3
The next longest stage was the point cloud generation one. This stage is composed by 3 step:
    - init point cloud;
    - project 2D to 3D (pinhole projection equation);
    - adjust to system;
I chose to bring only the last two step on GPU since the first one was already efficient on CPU.

Phase 4
I dealt with the last remain stage, the pre processing one. I decided to leave the main functionality of this stage
(super resolution) on cpu by creating namespace and structures GPU ready.

Phase 5
Revising the post processing stage in order to improve the psnr factor.
Optimization of the point cloud generation stage in order to improve the time performance.

Phase 6
I think how to implement the part two of the project: generating a plenoptic image starting from a multiview dataset
(left and right view of the same scene).

## Details

                                                ---Phase 1---
Now in ```main.cpp``` we can choose to perform between the previous pipeline and a new entire GPU pipeline based on
```config.entirePipelineGPU```.
To do so I added a public method ```initialize()``` inside ```Pipeline.hpp``` and I implemented it in ```Pipeline.cpp```
calling ```createDefaultStages()```. Then I created ```PipelineGPU.hpp``` where I defined the scheme of a subclass of 
```Pipeline.cpp``` that overrides the method ```createDefaultStages```. And in the new ```PipelineGPU.cpp``` I implemented
constructor and the new stage pipeline: PreProcessing -> PointCloudGeneration -> PlenopticRendering -> PostProcessingGPU.

                                                ---Phase 2---
For the new Post Processing stage on GPU I create the following files:
- ```PostProcessingGPU.hpp``` -> declares CPU stage wrapping GPU calls;
- ```PostProcessingGPU.cpp``` -> implements ```setupSteps()```, ```crackFilteringGPU()```, ```rotateMicroimagesGPU()``` calling GPU functions;
- ```PostProcessingGPU.cuh``` -> GPU interface for kernels;
- ```PostProcessingGPU.cu``` -> implementation of CUDA kernel.
In particular both kernels here use uchar4 for RGBA memory layout, cudaMemcpy to/from device and kernel launch is 2D grid of 16x16 blocks (256 threads per block). In the Crack Filtering Kernel I set:
- median filter per microimage (matches CPU cv::medianBlur behavior);
- one CUDA thread per pixel;
- direct global memory access using a buffer.
Instead for the Microimage 180° Rotation Kernel:
- each thread rotates one pixel;
- local coordinates computed per microimage, then mapped to global memory;
- deals correctly with arbitrary image size and square microimages based on MLA pitch parameters.

NOTES: evaluating the psnr factor between the ground truth plenoptic image (the one we get with the initial existing pipeline)
and the one we get here with this stage on GPU we achieve a psnr factor ~34.
FUTURE OPTIMIZATION: optimize microimage rotation kernel further (shared memory, coalesced access)...

                                                ---Phase 3---
For the new Post Processing stage on GPU I create the following files:
- ```PointCloudGenerationGPU.hpp``` -> declares CPU stage wrapping GPU calls;
- ```PointCloudGenerationGPU.cpp``` -> implements ```setupSteps()```, ```initPointCloudGPU()```, ```project2Dto3DGPU()``` and ```adjustPointCloudToSystemGPU``` calling GPU functions (except for ```initPointCloudGPU()`` tha uses the cpu version);
- ```PointCloudGenerationGPU.cuh``` -> GPU interface for kernels;
- ```PointCloudGenerationGPU.cu``` -> implementation of CUDA kernel. In particular  I used CUDA kernels and Thrust for parallelization. Kernels included: computeMaskKernel for flaging valid points based on depth threshold, 
scatterKernel for compacting valid points using inclusive scan and  adjustKernel for appling display scaling, offsets, and CDP/MLA modes in parallel (the first two kernels allow to parallelize the sequential dependence attendant in the CPU 
valid filtering procedure).
- ```GPUTypes.cuh``` -> defines ```RGB8``` struct to replace ```cv::Vec3b``` for GPU memory safety. Ensures correct 3-byte alignment for colors on the device.

NOTES: Despite multiple optimization attempts on the GPU implementation of the point cloud generation stage, the psnr value could not be improved beyond ~31. I tested several approaches, including enforcing numerical consistency with the CPU version (scaling factors, float vs. double precision), ensuring stable ordering of points (avoiding atomic operations and using thrust-based compaction), and aligning the projection pipeline exactly with the CPU logic. Additionally, post-processing was verified independently and shown to achieve higher psnr when isolated, suggesting that the error originates in the point cloud generation stage. However, none of these modifications resulted in a measurable psnr improvement.

                                                ---Phase 4---
Originally, the idea was upscaling RGB and depth images to the GPU using OpenCV’s CUDA functions (```cv::cuda::GpuMat``` and ```cv::cuda::resize```) to gain speed. Hozever OpenCV prebuilt libraries on Windows often don’t include ```cv::cuda::resize```, which caused the build errors. So I created ```GPU::PreProcessing``` namespace.
I implemented Pre Processing GPU pipeline stage, with methods like ```superResolutionGPU()``` and ```setupSteps()```.
This ensures future CUDA kernels can be added without changing the interface or pipeline integration.
So basically the current ```cv::resize``` is still CPU-based.

NOTES: As a result, the observed runtime of this new stage decreased to ~30ms, even without moving the resize computation to the GPU. aybe because here now avoid unnecessary copies by structuring the stage like a GPU stage. In addition we have possibly reuse already allocated memory in ```PipelineData``` (rgbImage / depthMap) instead of creating extra temporaries. 

FUTURE OPTIMIZATION: true GPU-based resizing using ```cv::cuda::GpuMat``` and ```cv::cuda::resize``` could further reduce the runtime for very large images;


                                                ---Phase 5---
The original version of the GPU Post Processing Stage, which achieved a PSNR of approximately 33.9, relied on a simpler median computation and a more implicit handling of boundaries and crack masking. For the updated implementation I  introduced a region-based (ROI) processing strategy using explicit microimage descriptors on the GPU. Additionally I restructured the kernel to operate per microimage and avoid out-of-bounds accesses.
Further refinements included removing the alpha channel from the median computation (to preserve crack masking consistency), introducing device-side median selection (later reverted to full sorting for correctness), and initializing the output buffer to avoid undefined writes. Despite these improvements, the PSNR dropped to around 31.18 and remained stable across multiple adjustments. This let me think to have achieve some psnr roofline for such implementations. 
*The main source of this gap is currently attributed to differences in median filtering behavior (especially ordering and tie handling) and possibly subtle mismatches in border handling or mask application.*

Regarding the time improvance of the Point Cloud Generation Stage the first step was removed the big cost given by "cuda malloc
+ cuda free" for each call. So I replaced this memory implementation in the ```project2Dto3D()``` host function with a new one 
"allocate one time + resue" by making static buffers static and reusable. 
Only by applying this first step the time performance for this stage drop significantly: from ~190ms to ~140ms (mean of the last
5 launches in a block of 10 runs).
Then I tried other options like merging the mask and the projection kernels into only one kernel, and also reduce the number of memcpy by packing (X, Y, Z, px, py, colors) into a struct (ending with only one memcpy calls) but I gained only fewers milliseconds. So doing I achieved a mean time of ~130ms. In addition I marked the pointer in the kernel as ```__restrict__``` in order to say to the compiler to don't overlap these pointers and activate the caching. Time performance doesn't improve anymore.
*This might means the computation wasn't the problem, but memory allocation was. Now we are memory-bound*. 

                                                ---Phase 6---
Here the precise assignement consists of merging point clouds from three cameras while preserving geometric consistency and reducing duplicates.
I thought about the following roadmap:
    - get position, rotation, and intrinsics for each camera (```CameraCalibration.hpp``` and ```CameraCalibration.cpp```);
    - convert RGB + depth into a 3D point cloud for each camera by means ```PointCloudGenerationGPU::project2Dto3D```;
    - merge point clouds from the previous step into the central camera’s coordinate system (camera 3). I use the rigid body transformation liked in the email from the supervisor:

                                                    P_world​ = R ⋅ P_camera​ + T

        where P_camera := 3D point in the local camera world
                  R    := 3x3 rotation matrix from Euler angles in ```rotation_xyz_deg```
                  T    := 3D translation vector from ```position_mm```
    - concatenate points from all cameras into a single cloud;
    - eliminate points that are at the same position within a small tolerance (e.g., 1e-5) like Brenno suggested;
    Possible algorithms? Do I need to scan always each points??
    - re-run post processing stage?