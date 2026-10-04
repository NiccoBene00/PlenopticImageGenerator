
## Notes taken during the project
---

**FIRST PART: GPU PIPELINE**
												
Phase 1:
I create a toggle system that is able to switch pipeline mode (previuous mode where only the rendering stage was on GPU and 
the new one where we aim to bring each stage on GPU).

Phase 2:
I analyze how bringing everything on GPU. First I observe that the most expensive stage is the Post Processing one (above 50% of the total stage execution time), so I focus on creating a GPU Post Processing stage.

Phase 3:
The next longest stage was the point cloud generation one. This stage is composed by 3 step:
    - init point cloud;
    - project 2D to 3D (pinhole projection equation);
    - adjust to system;
I chose to bring only the last two step on GPU since the first one was already efficient on CPU.

Phase 4:
I dealt with the last remain stage, the pre processing one. I decided to leave the main functionality of this stage
(super resolution) on cpu by creating namespace and structures GPU ready.

Phase 5:
Revising the post processing stage in order to improve the psnr factor.
Optimization of the point cloud generation stage in order to improve the time performance.


**SECOND PART: MULTI-VIEW GPU PIPELINE**

Phase 1:
I think how to implement the part two of the project: generating a plenoptic image starting from a multiview dataset
(left and right view of the same scene).

Phase 2:
I created a Camera Calibration class to store extrinsic and intrinsic parameters for multiple cameras from a json file. 

Phase 3:
I implemented a new Dataset Loader, since now I need to manage different datasets (each of which based on different rgb scene and
depth map).

Phase 3:
I created a new pipeline (MultiView GPU) by adding a new stage, the Multi View Point Cloud.

Phase 4:
I extendent the MultiView GPU pipeline by creating the next GPU-based stage: MultiView Registration.

Phase 5:
I yield the next stage: MultiView Registration in order to apply the rigid body transformation and merge the differents point clouds
into a unified one.



## Implementation Details
---

**FIRST PART**

### Phase 1

Now in ```main.cpp``` we can choose to perform between the previous pipeline and a new entire GPU pipeline based on
```config.entirePipelineGPU```.
To do so I added a public method ```initialize()``` inside ```Pipeline.hpp``` and I implemented it in ```Pipeline.cpp```
calling ```createDefaultStages()```. Then I created ```PipelineGPU.hpp``` where I defined the scheme of a subclass of 
```Pipeline.cpp``` that overrides the method ```createDefaultStages```. And in the new ```PipelineGPU.cpp``` I implemented
constructor and the new stage pipeline: PreProcessing -> PointCloudGeneration -> PlenopticRendering -> PostProcessingGPU.


### Phase 2

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


### Phase 3

For the new Post Processing stage on GPU I create the following files:
- ```PointCloudGenerationGPU.hpp``` -> declares CPU stage wrapping GPU calls;
- ```PointCloudGenerationGPU.cpp``` -> implements ```setupSteps()```, ```initPointCloudGPU()```, ```project2Dto3DGPU()``` and ```adjustPointCloudToSystemGPU``` calling GPU functions (except for ```initPointCloudGPU()`` tha uses the cpu version);
- ```PointCloudGenerationGPU.cuh``` -> GPU interface for kernels;
- ```PointCloudGenerationGPU.cu``` -> implementation of CUDA kernel. In particular  I used CUDA kernels and Thrust for parallelization. Kernels included: computeMaskKernel for flaging valid points based on depth threshold, 
scatterKernel for compacting valid points using inclusive scan and  adjustKernel for appling display scaling, offsets, and CDP/MLA modes in parallel (the first two kernels allow to parallelize the sequential dependence attendant in the CPU 
valid filtering procedure).
- ```GPUTypes.cuh``` -> defines ```RGB8``` struct to replace ```cv::Vec3b``` for GPU memory safety. Ensures correct 3-byte alignment for colors on the device.

NOTES: Despite multiple optimization attempts on the GPU implementation of the point cloud generation stage, the psnr value could not be improved beyond ~31. I tested several approaches, including enforcing numerical consistency with the CPU version (scaling factors, float vs. double precision), ensuring stable ordering of points (avoiding atomic operations and using thrust-based compaction), and aligning the projection pipeline exactly with the CPU logic. Additionally, post-processing was verified independently and shown to achieve higher psnr when isolated, suggesting that the error originates in the point cloud generation stage. However, none of these modifications resulted in a measurable psnr improvement.


### Phase 4

Originally, the idea was upscaling RGB and depth images to the GPU using OpenCV’s CUDA functions (```cv::cuda::GpuMat``` and ```cv::cuda::resize```) to gain speed. Hozever OpenCV prebuilt libraries on Windows often don’t include ```cv::cuda::resize```, which caused the build errors. So I created ```GPU::PreProcessing``` namespace.
I implemented Pre Processing GPU pipeline stage, with methods like ```superResolutionGPU()``` and ```setupSteps()```.
This ensures future CUDA kernels can be added without changing the interface or pipeline integration.
So basically the current ```cv::resize``` is still CPU-based.

NOTES: As a result, the observed runtime of this new stage decreased to ~30ms, even without moving the resize computation to the GPU. aybe because here now avoid unnecessary copies by structuring the stage like a GPU stage. In addition we have possibly reuse already allocated memory in ```PipelineData``` (rgbImage / depthMap) instead of creating extra temporaries. 

FUTURE OPTIMIZATION: true GPU-based resizing using ```cv::cuda::GpuMat``` and ```cv::cuda::resize``` could further reduce the runtime for very large images;


### Phase 5

The original version of the GPU Post Processing Stage, which achieved a PSNR of approximately 33.9, relied on a simpler median computation and a more implicit handling of boundaries and crack masking. For the updated implementation I  introduced a region-based (ROI) processing strategy using explicit microimage descriptors on the GPU. Additionally I restructured the kernel to operate per microimage and avoid out-of-bounds accesses.
Further refinements included removing the alpha channel from the median computation (to preserve crack masking consistency), introducing device-side median selection (later reverted to full sorting for correctness), and initializing the output buffer to avoid undefined writes. Despite these improvements, the PSNR dropped to around 31.18 and remained stable across multiple adjustments. This let me think to have achieve some psnr roofline for such implementations. 
*The main source of this gap is currently attributed to differences in median filtering behavior (especially ordering and tie handling) and possibly subtle mismatches in border handling or mask application.*

Regarding the time improvance of the Point Cloud Generation Stage the first step was removed the big cost given by "cuda malloc with
cuda free" for each call. So I replaced this memory implementation in the ```project2Dto3D()``` host function with a new one 
"allocate one time + resue" by making static buffers static and reusable. 
Only by applying this first step the time performance for this stage drop significantly: from ~190ms to ~140ms (mean of the last
5 launches in a block of 10 runs).
Then I tried other options like merging the mask and the projection kernels into only one kernel, and also reduce the number of memcpy by packing (X, Y, Z, px, py, colors) into a struct (ending with only one memcpy calls) but I gained only fewers milliseconds. So doing I achieved a mean time of ~130ms. In addition I marked the pointer in the kernel as ```__restrict__``` in order to say to the compiler to don't overlap these pointers and activate the caching. Time performance doesn't improve anymore.
*This might means the computation wasn't the problem, but memory allocation was. Now we are memory-bound*. 



**SECOND PART**

### Phase 1

Here the precise assignement consists of merging point clouds from three cameras while preserving geometric consistency and reducing duplicates.

I thought about the following roadmap:

- get position, rotation, and intrinsics for each camera (```CameraCalibration.hpp``` and ```CameraCalibration.cpp```);
- convert RGB + depth into a 3D point cloud for each camera by means ```PointCloudGenerationGPU::project2Dto3D```;
- merge point clouds from the previous step into the central camera’s coordinate system (camera 3). I use the rigid body transformation liked in the email from the supervisor:

```text
P_world​ = R ⋅ P_camera​ + T
```

where:

```text
P_camera := 3D point in the local camera world
R        := 3x3 rotation matrix from Euler angles in rotation_xyz_deg
T        := 3D translation vector from position_mm
```

- concatenate points from all cameras into a single cloud;
- eliminate points that are at the same position within a small tolerance (e.g., 1e-5) like Brenno suggested; (Possible algorithms? Do I need to scan always each points??)
- re-run post processing stage


### Phase 2
Each camera contains its 3D position in millimeters, rotation in Euler angles (XYZ, degrees), focal length in pixels, and principal point. The class (```CameraCalibration.hpp``` and ```CameraCalibration.cpp```) parses the JSON, stores the data in a struct, and provides methods to retrieve the translation vector and rotation matrix for each camera. This ìmakes sure that each point cloud generated from a camera can be accurately transformed into a common global coordinate system.


### Phase 3

I created e new hpp file (```MultiViewDatasetLoader.hpp```) to load a multi-view dataset from a folder and prepares all the data
needed for the reconstruction following pipeline. Here the main steps of this file are:
    1. Validate dataset folder
    2. Load camera calibration (extrinsics + intrinsics)
    3. Initialize per-view point cloud containers
    4. Store dataset parameters (camera intrinsics, depth format, etc.)


### Phase 4

I introduced ```MultiViewPointCloud.hpp```/```MultiViewPointCloud.cpp```. This phase is about just re-using the ```project2Dto3D()``` function in order to convert each depth map into a 3D point cloud. Basically here For each camera:
    1. Load RGB image and depth map
    2. Build a structure point cloud: (px, py, depth, color)
    3. Send data to GPU
    4. Backproject pixels into 3D coordinates (X, Y, Z)
    5. Store resulting 3D point cloud in an array of point cloud
    


### Phase 5

I created a new cu stage called ```MultiViewRegistration``` which kernels run completely on GPU, where first I computed the rigid body transformation by following this logic: 

Input:
- (X, Y, Z): point in camera coordinates
- R: 3x3 rotation matrix
- t: translation vector

Output:
- transformed point in world coordinates

Formula:
p_world = R^(-1) * p_cam + C

*Notes about the revise formule used for the rigi body transformation*: every reconstructed point cloud is initially expressed in camera coordinates rather than in a shared world reference frame. Hence we know that a rigid transformation is required to align all point clouds before merging and plenoptic rendering. Generally camera extrinsics are commonly represented using the world-to-camera formulation

                                                    p_camera = R * p_world + t

where R is the rotation matrix and t is the translation vector. To recover world coordinates from camera-space points, we need to use the inverse transformation. Starting from the previous equation:

                                                    p_camera - t = R * p_world
                                                    R^(-1) * (p_camera - t) = p_world
                                                    p_world = R^(-1) * (p_camera) - R^(-1) * t
	​
However, in the adopted calibration setup, the dataset directly stores the camera center position C in world coordinates rather than the classical extrinsic translation vector t=−RC. As a result, the implemented transformation becomes:

                                                    p_world = R^(-1) * p_camera + C

where the inverse rotation aligns the reconstructed camera-space points with the global reference frame, while the camera center translates them into the correct world position.

Hence at the end of the day I implemented the Merge step:
    1. Concatenate all transformed point clouds
    2. Optionally remove duplicates (currently todo)
    3. Build final unified point cloud

before to pass give forward everything to the Plenoptic Rendering stage.



## Command to run the project in multi-view
---

For now the design implemented in the ```main.cpp```  provides a mechanism to detect if the path-to-dataset points
to a folder with multiple .json/.exr files. In the positive case the system switch to the multi-view pipeline.
This design should be revised and optmized.
The command for the multi-view becomes:

```bash
C:\path-to-project\code\out\build\x64-release\pig.exe --system_spec <path-to-system-spec>.json --dataset <path-to-multi-view-dataset> --config <path-to-config>.json --output <path-to-output>.png
```
