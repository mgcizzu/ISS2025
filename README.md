The code in this repository is the first major update of the code from Lee et al, 2024. 

The original code is found at https://github.com/Moldia/Lee_2023 but there was a need for a major update, moving away from old and unmaintained packages, incorporating new features etc...
This repository is a heavily modified and updated fork of the original code. We chose to maintain a frozen version of the Lee_2023 repository for reproducibility purposes, but we encourage everybody to update to this latest version from this repository.

A quick glance at the updates:

# general things
All the code has been updated so to run on packages that are excluded from the anaconda official channels. This is a consequence of some institutions forbidding the use of anaconda because of licensing issues.

The code now runs exclusively on packages available on conda-forge (with the exception of the deconvolution module, that requires nvidia packages). nvidia channels are fine.

All the installations .yml files have been simplified trying to unpin the versions from as many packages as possible and let the pip dependency resolver to do the job. I have left pinned only the packages with a strict requirement on version.

All the modules have been updated, whenever possible, with the latest available version of every package. The code has been modified accordingly, to solve issues arising because of commands deprecated in these new package version (ie, big drama with pandas.append).

There were many little mistakes in the example notebooks. They were mostly harmless but were propagating small errors in the analysis if we didn't carefully check what we were doing. Or they were giving import errors because of typos, etc...
They are as fixed as I could.

Here's some more specific info about the individual bits of code.


# preprocessing module:
The preprocessing module is now able to read, sort and maximum project directly from .lif files (both auto-saved or exported) from the Leica LasX software. Refer to the example notebooks for informations about how to run the new function.

# deconvolution module:
The image deconvolution module based on flowdec has been deprecated. A new one has been implemented, based on RedLionFish (https://github.com/rosalindfranklininstitute/RedLionfish). This is because flowdec was not maintained actively and worked on a prehistoric version of Tensorflow creating all sorts of compatibility issues.

The deconvolution module now includes a function to deconvolve and project images directly from .lif files. Please refer to the manual for further info.

New things will be added soon (scroll down).

# decoding module:
The major update in the decoding module is the enabling of the *dense mode*. This new modality allows (with a considerable computation time cost) to perform spot detection in individual channels, while the old modality enforced a pseudoanchor detection, producing a spot underestimation in crowded images. Please refer to the notebook for details.

# postprocessing module:
The way Cellpose was set up in our original functions was defaulting to CPU. I think this is because our code was so old we didn't really use GPUs at the time. Now the function is updated to default to GPU use whenever available.
I split the notebooks and created new ones in a more modular structure for didactic purposes. Now there is a specific notebook for each segmentation tool available (Cellpose or Stardist), plus a notebook that allows to inspect the segmentation mask over DAPI, or even to export the mask for TissUUmaps visualization.

# What is missing
I have written mipping and deconvolution functions to parse .nd2 files from Nikon microscopes. However they have been written around a single example file and they are not rigorously tested. They are commented out but they can be tested/used if needed from the respective .py scripts.

We are working on functions to incorporate alternative deconvolution methods (ie. Deconwolf). They will soon be added to the code.

It would be great to include functions to deal with unusual cases that we have surely worked with in the past but we haven't propagated the knowledge for. 

Here's a preliminary list:

- Notebooks to create anndata object after Ilastik segmentation
- Notebooks for Protein / EdU quantification and thresholding
- Notebooks for segmentation-free approaches
- Notebooks for spatial domain analysis (ie Banksy)

Please feel free to donate these notebooks if you have them at hand.

Please test and report problems as issues via GitHub.
