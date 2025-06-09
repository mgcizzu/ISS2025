# --- Standard Library ---
import os
from os import listdir
from os.path import isfile, join
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

# --- Third-Party Libraries ---
import numpy as np
import pandas as pd
import tifffile
import dask
import dask.array as da
from tqdm import tqdm
from aicspylibczi import CziFile
from readlif.reader import LifFile

# --- Custom Modules ---
import RedLionfishDeconv as rl
import ISS_deconvolution.psf as fd_psf




'''
#THIS IS AN EXAMPLE OF PSF DATA
PSF_metadata = {'na':0.8,
'm':20,
'ni0':1.42,
'res_lateral':0.419,
'res_axial':1.718,
 'channels':{
 'AF750':{
    'wavelength':.773},
  'Cy5':{
    'wavelength':.673},
  'Cy3':{
    'wavelength':.561},
  'AF488':{
    'wavelength':.519},
 'DAPI':{
    'wavelength':.465}
 }
}
m= magnification
ni0 = refraction coefficient of the immersion medium
res_lateral = resolution in xy
res_axial = resolution in z
'''

def custom_copy(src, dest):
    """Custom function to copy a file to a destination."""
    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(src))
    shutil.copyfile(src, dest)
    

def generate_psf(psf_output, resxy, resz, wavelength, NA, ni):
    # dw_bw command to generate PSF
    command = [
        "dw_bw",  # Make sure dw_bw is in your PATH or specify the full path
        "--resxy", str(resxy),  # Lateral pixel size (nm)
        "--resz", str(resz),    # Axial pixel size (nm)
        "--lambda", str(wavelength),  # Wavelength (nm)
        "--NA", str(NA),  # Numerical aperture
        "--ni", str(ni),  # Refractive index
        psf_output  # Output PSF file (e.g., PSF_dapi.tif)
    ]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        #print(f"PSF generated and saved as {psf_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating PSF: {e}")


def deconvolve_image(input_image, psf_image, output_image, iterations, tilesize=None):
    # DeconWolf command to deconvolve the image

    command = [
    "deconwolf",
    "--iter", str(iterations),
    input_image,
    psf_image,
    "--out", output_image
    ]

    if tilesize is not None:
        command += ['--tilesize', str(tilesize)]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        print(f"Deconvolution finished. Output saved to {output_image}")
        
    except subprocess.CalledProcessError as e:
        print(f"\033[91mError during deconvolution: {e}\033[0m")
    except FileNotFoundError:
        print(f"\033[91mError: DeconWolf executable not found at {deconwolf_path}\033[0m")
    except Exception as e:
        print(f"\033[91mUnexpected error: {e}\033[0m")


    


# -------------------------------------------------------------------------------------
# CZI
# -------------------------------------------------------------------------------------

def deconvolve_czi(input_file, outpath, image_dimensions=[2048, 2048], PSF_metadata=None, chunk_size=None,  mip=True, cycle=0, tile_size_x=2048, tile_size_y=2048):

    """
    Process CZI files, deconvolve the image stacks, apply maximum intensity projection (if specified), 
    and create an associated XML with metadata.
    
    Parameters:
    - input_file: Path to the input CZI file.
    - outpath: Directory where the processed images and XML will be saved.
    - chunk_size= [x,y] where x and y are the size of the chunks the image needs to be cut into for small GPU processing
    - mip: Boolean to decide whether to apply maximum intensity projection. Default is True.
    - cycle: Int to specify the cycle number. Default is 0.
    - tile_size_x: Size of the tile in X dimension. Default is 2048.
    - tile_size_y: Size of the tile in Y dimension. Default is 2048.
    
    Returns:
    - A string indicating that processing is complete.
    """

    def cropND(img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(lambda a, da: a+da, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

    # Load the CZI file and retrieve its dimensions.
    czi = aicspylibczi.CziFile(input_file)
    dimensions = czi.get_dims_shape() 
    chsize = dimensions[0]['C'][1]
    msize = dimensions[0]['M'][1]
    z_size=dimensions[0]['Z'][1]

# Create the output directory if it doesn't exist.
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # Check if mip is and cycle is not zero.
    if cycle != 0:
        Bxcoord = []
        Bycoord = []
        Btile_index = []
        filenamesxml = []
        Bchindex = []
        
        psf_dict = {}
        for idx, channel in enumerate(sorted(PSF_metadata['channels'])):
            psf_dict[idx] = fd_psf.GibsonLanni(
                na=float(PSF_metadata['na']),
                m=float(PSF_metadata['m']),
                ni0=float(PSF_metadata['ni0']),
                res_lateral=float(PSF_metadata['res_lateral']),
                res_axial=float(PSF_metadata['res_axial']),
                wavelength=float(PSF_metadata['channels'][channel]['wavelength']),                
                size_x=tile_size_x,
                size_y=tile_size_y,
                size_z=z_size  # Use the Z dimension from the CZI file
            ).generate()

        # Process each channel and mosaic tile
        for m in tqdm(range(0, msize)):
            for ch in range(0, chsize):
                # Read image data for the current mosaic tile and channel
                img, shp = czi.read_image(M=m, C=ch)
                img=np.squeeze(img, axis=(0,1,2,4))
                
                # Read metadata for the current mosaic tile and channel

                meta = czi.get_mosaic_tile_bounding_box(M=m, Z=0, C=ch)
           

                # Instantiate the Richardson-Lucy deconvolution algorithm
                algo = RichardsonLucyDeconvolver(img.ndim, pad_mode="2357", pad_min=(6,6,6))

                # Check if chunk_size is provided
                if chunk_size:
                    # Define chunk dimensions
                    chunked_dims = (z_size, chunk_size[0], chunk_size[1])

                    # Convert image data to a chunked dask array
                    arr = da.from_array(img, chunks=chunked_dims)
                    cropped_kernel = cropND(psf_dict[ch], chunked_dims)

                    # Define deconvolution function for chunks
                    def deconv(chunk):
                        tmp = algo.initialize().run(fd_data.Acquisition(data=chunk, kernel=cropped_kernel), 50)
                        return tmp.data 

                    # Apply chunked deconvolution
                    deconvolved = arr.map_overlap(deconv, depth=(6,6,6), boundary='reflect', dtype='uint16').compute(num_workers=1)
                else:
                    # Regular deconvolution for the entire image
                    deconvolved = algo.initialize().run(fd_data.Acquisition(data=img, kernel=psf_dict[ch]), 50)
                if chsize != len(PSF_metadata['channels']):
                    raise ValueError("Mismatch between CZI file channels and PSF_metadata channels.")
                #print(deconvolved.data.shape)
                # Check if mip (max intensity projection) is enabled
                if mip:
                    processed_img = np.max(deconvolved.data, axis=0).astype('uint16')
                else:
                    processed_img = deconvolved.data.astype('uint16')
            
                # Construct filename for the processed image
                n = str(0)+str(m+1) if m < 9 else str(m+1)
                filename = f'Base_{cycle}_c{ch+1}m{n}_ORG.tif'

                # Save the processed image
                tifffile.imwrite(os.path.join(outpath, filename), processed_img)
            # Append metadata to the placeholders.
            Bchindex.append(ch)
            Bxcoord.append(meta.x)
            Bycoord.append(meta.y)
            Btile_index.append(m)
            filenamesxml.append(filename)
 # Adjust the XY coordinates to be relative.
        nBxcord = [x - min(Bxcoord) for x in Bxcoord]
        nBycord = [y - min(Bycoord) for y in Bycoord]
        
        # Create a DataFrame to organize the collected metadata.
        metadatalist = pd.DataFrame({
            'Btile_index': Btile_index, 
            'Bxcoord': nBxcord, 
            'Bycoord': nBycord, 
            'filenamesxml': filenamesxml,
            'channelindex': Bchindex
        })
        
        metadatalist = metadatalist.sort_values(by=['channelindex','Btile_index'])
        metadatalist.reset_index(drop=True)

        # Initialize the XML document structure.
        export_doc = ET.Element('ExportDocument')
        
        # Populate the XML document with metadata.
        for index, row in metadatalist.iterrows():
            image_elem = ET.SubElement(export_doc, 'Image')
            filename_elem = ET.SubElement(image_elem, 'Filename')
            filename_elem.text = row['filenamesxml']
            
            bounds_elem = ET.SubElement(image_elem, 'Bounds')
            bounds_elem.set('StartX', str(row['Bxcoord']))
            bounds_elem.set('SizeX', str(tile_size_x))
            bounds_elem.set('StartY', str(row['Bycoord']))
            bounds_elem.set('SizeY', str(tile_size_y))
            bounds_elem.set('StartZ', '0')
            bounds_elem.set('StartC', '0')
            bounds_elem.set('StartM', str(row['Btile_index']))
            
            zoom_elem = ET.SubElement(image_elem, 'Zoom')
            zoom_elem.text = '1'

        
        # Save the constructed XML document to a file.
        xml_str = ET.tostring(export_doc)
        with open(outpath + 'Base_' + str(cycle) + '_info.xml', 'wb') as f:
            f.write(xml_str)
  
    return "Processing complete."
    


# -------------------------------------------------------------------------------------
# LEICA EXPORTED + AUTOSAVED
# -------------------------------------------------------------------------------------

def deconvolve_leica(input_dir, 
                     output_dir_prefix, 
                     cycle,
                     deconvolution_method,
                     image_dimensions=[2048, 2048], 
                     PSF_metadata=None, 
                     chunk_size=None, 
                     mip=True,
                     mode='autosaved'):
    """
    Process the images from the given directories.

    Parameters:
    - input_dir: List of directories containing the images to process.
    - output_dir_prefix: Prefix for the output directories.
    - cycle: Number of ISS cycle to be processed.
    - deconvolution_method: 'redlionfish' (gpu) or 'deconwolf' (cpu)
    - image_dimensions: Dimensions of the images (default: [2048, 2048]).
    - PSF_metadata: Metadata for Point Spread Function (PSF) generation.
    - chunk_size [x,y]: Size of chunks for processing. If None, the entire image is processed.
                  Small GPUs will require chunking. Enable if you run out of gRAM
    - mip: Boolean to decide whether to apply maximum intensity projection. Default is True. 
           If mip=false the stack is deconvolved but saved as an image stack without projecting it
    - mode='autosaved', ='exported' if exported via the export function in LasX

    Returns:
    None. Processed images are saved in the output directories.
    """
    
    script_start_time = time.time()
    
    # ----- Step 1: Initial validation and input preparation -----
    # Ensure PSF metadata is provided. Normalize input directory names.
    # Print initial info banner for user feedback.
    
    if PSF_metadata is None:
        raise ValueError("PSF_metadata is required to generate PSF.")

    if mode is None:
        raise ValueError("Leica saving mode must be specified as either 'autosaved' or 'exported'.")

    
    print("\033[96m\033[1m" + "="*60 + "\033[0m")
    print("\033[96m\033[1m" + 
          ("Deconvolving Leica files from exported mode" if mode == 'exported' else
           "Deconvolving Leica files from autosaved mode" if mode == 'autosaved' else
           "Deconvolving Leica files") + 
          "\033[0m")
    print("\033[96m\033[1m" + "="*60 + "\033[0m")

    print("\033[1;96m>> Using Deconvolution method: {} <<\033[0m".format(
    "Deconwolf" if deconvolution_method == "deconwolf"
    else "RedLionFish" if deconvolution_method == "redlionfish"
    else f"Unknown ({deconvolution_method})"))

    base = cycle
    print(f"\033[1;90mProcessing Cycle {base}\033[0m")
    
    input_dir = input_dir.replace("%20", " ")
    
    # ----- Step 2: Process input directories (bases/cycles) -----
    # Process each input folder corresponding to a cycle. Extract relevant TIF files,
    # identify regions, and prepare region numbers based on naming convention.
    
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif') and 'dw' not in f and '.txt' not in f]

    if mode == 'autosaved':
        regions = sorted(set(f.split('--')[0] for f in tif_files))
        region_numbers = sorted(set(re.search(r'Region\s*(\d+)', r).group(1) for r in regions))
    elif mode == 'exported':
        regions = sorted(set(re.search(r'(Region\s*\d+)', f).group(1) for f in tif_files))
        region_numbers = sorted(set(re.search(r'Region\s*(\d+)', r).group(1) for r in regions))

    print("Regions to be processed:", regions)

    # ----- Step 3: Loop over regions -----
    # For each region in the current base, filter TIFs, extract tile list,
    # and prepare output directories.

    for region_index, region in enumerate(regions):
        print(f"\033[1;90mProcessing Region {region_index + 1}/{len(regions)}\033[0m")
        filtered_tifs = [f for f in tif_files if region in f]
        
        if mode == 'autosaved':
            tiles = pd.Series(filtered_tifs).str.split('--').str[1].str.extract(r'(\d+)')[0].dropna().unique()
        elif mode == 'exported':
            tiles = pd.Series(filtered_tifs).str.extract(r'_s(\d+)_')[0].dropna().unique()
        tiles = sorted(tiles, key=int)

        print('Sorted tiles:', tiles)

        # ----- Step 4: Set up directory structure for outputs -----
        # Create structured folders for saving mipped and deconvolved outputs.

        if len(regions) == 1:
            output_directory = output_dir_prefix
        else:
            output_directory = f"{output_dir_prefix}_R{region_numbers[region_index]}"

        mipped_directory = os.path.join(output_directory, 'preprocessing', 'mipped')
        os.makedirs(mipped_directory, exist_ok=True)

        base_directory = os.path.join(mipped_directory, f'Base_{base}')
        os.makedirs(base_directory, exist_ok=True)

        # ----- Step 5: Copy metadata if available -----
        # If the metadata directory contains files for the current region, copy them.

        print('Extracting metadata')
        metadata_dir = os.path.join(input_dir, 'Metadata')
        metadata_files = [f for f in os.listdir(metadata_dir) if region in f]
        metadata_file = [f for f in metadata_files if 'properties' not in f][0]

        if metadata_files:
            os.makedirs(os.path.join(base_directory, 'MetaData'), exist_ok=True)
            custom_copy(
                os.path.join(metadata_dir, metadata_file),
                os.path.join(base_directory, 'MetaData')
            )

        # =============================== RedLionFish Deconvolution ===============================
        if deconvolution_method == 'redlionfish':
        
            # ----- Step 6: Generate PSFs for all channels -----
            # Calculate PSF size from a sample tile and create PSFs for each channel
        
            print("Calculating the PSF...")
            if mode == 'autosaved':
                sample_tile = [f for f in filtered_tifs if f"--Stage00--" in f]
            elif mode == 'exported':
                sample_tile = [f for f in filtered_tifs if f"_s0_" in f]
        
            size_z = int(len(sample_tile) / len(PSF_metadata['channels']))
        
            psf_dict = {}
            for channel in PSF_metadata['channels']:
                psf_dict[channel] = fd_psf.GibsonLanni(
                    na=float(PSF_metadata['na']),
                    m=float(PSF_metadata['m']),
                    ni0=float(PSF_metadata['ni0']),
                    res_lateral=float(PSF_metadata['res_lateral']),
                    res_axial=float(PSF_metadata['res_axial']),
                    wavelength=float(PSF_metadata['channels'][channel]['wavelength']),
                    size_x=image_dimensions[0],
                    size_y=image_dimensions[1],
                    size_z=size_z
                ).generate()
        
            # ----- Step 7: Deconvolve each tile and channel -----
            # Stack z-planes, deconvolve with RedLionFish
                        
            for tile in tqdm(sorted(tiles, key=int)):
                if mode == 'autosaved':
                    tile_files = [f for f in filtered_tifs if f"--Stage{tile}--" in f]
                elif mode == 'exported':
                    tile_files = [f for f in filtered_tifs if f"_s{tile}_" in f]
        
                for channel in sorted(PSF_metadata['channels']):
                    print(f"\033[90m[Cycle {base}] Tile {tile}, Channel {channel}...\033[0m")
                    tile_channel_start = time.time()
        
                    output_file_path = os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif')
        
                    if os.path.exists(output_file_path):
                        print(f"File {output_file_path} already exists. Skipping.")
                        continue
        
                    # ----- Step 7a: Stack z-planes into 3D image -----
                    if mode == 'autosaved':
                        channel_files = [f for f in tile_files if f"--C{str(channel).zfill(2)}" in f]
                    elif mode == 'exported':
                        channel_files = [f for f in tile_files if f"_ch{channel}" in f]
        
                    stacked_images = np.stack([
                        tifffile.imread(os.path.join(input_dir, f)) for f in channel_files
                    ])
        
                    # ----- Step 7b: Run RedLionFish deconvolution -----
                    
                    deconvolved_img = rl.doRLDeconvolutionFromNpArrays(
                        stacked_images, psf_dict[channel], niter=50)
        
                    # ----- Step 8: Post-process output (mip or full stack) -----
                    # Apply MIP if enabled, or keep full deconvolved stack.
                    if mip:
                        processed_img = np.max(deconvolved_img, axis=0).astype('uint16')
                    else:
                        processed_img = deconvolved_img.astype('uint16')
        
                    tifffile.imwrite(output_file_path, processed_img)
                    print(f"Mipped images saved in directory: {base_directory}")

                    tile_channel_end = time.time()
                    print(f"\033[1;37m[Timing] Full deconvolution cycle for Tile {tile}, Channel {channel} took {tile_channel_end - tile_channel_start:.2f} seconds\033[0m")


        # =============================== Deconwolf Deconvolution ===============================
        elif deconvolution_method == 'deconwolf':
            
            # ----- Step 6: Generate PSFs for all channels -----
            # Create PSFs using provided metadata for each channel.

            psf_filepath = os.path.join(base_directory, 'PSF')
            os.makedirs(psf_filepath, exist_ok=True)
            
            psf_dict = {}
            for channel, info in PSF_metadata['channels'].items():
                wavelength_nm = float(info['wavelength']) * 1000
                psf_filename = os.path.join(psf_filepath, f"PSF_channel_{channel}.tif")
                generate_psf(
                    psf_output=psf_filename,
                    resxy=PSF_metadata['res_lateral'] * 1000,
                    resz=PSF_metadata['res_axial'] * 1000,
                    wavelength=wavelength_nm,
                    NA=PSF_metadata['na'],
                    ni=PSF_metadata['ni0']
                )
                psf_dict[channel] = psf_filename
                    
            # ----- Step 7: Deconvolve each tile and channel -----
            # Stack z-planes for each tile and channel, then deconvolve using Deconwolf.
            
            for tile in tqdm(sorted(tiles, key=int)):
                if mode == 'autosaved':
                    tile_files = [f for f in filtered_tifs if f"--Stage{tile}--" in f]
                elif mode == 'exported':
                    tile_files = [f for f in filtered_tifs if f"_s{tile}_" in f]

                dw_tmp_dir = os.path.join(base_directory, 'deconwolf tmp')
                os.makedirs(dw_tmp_dir, exist_ok=True)
            
                for channel in sorted(PSF_metadata['channels']):
                    print(f"\033[90m[Cycle {base}] Tile {tile}, Channel {channel}...\033[0m")
                    tile_channel_start = time.time()
            
                    dw_output_dir = os.path.join(base_directory, 'stacked')
                    os.makedirs(dw_output_dir, exist_ok=True)
                    dw_output = os.path.join(dw_output_dir, f'Base_{base}_s{tile}_C0{channel}.tif')
                    output_file_path = os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif')
            
                    if os.path.exists(output_file_path):
                        print(f"File {output_file_path} already exists. Skipping.")
                        continue
            
                    # ----- Step 7a: Stack z-planes for each tile/channel -----
                    # Load all matching channel images for a tile, stack them into a 3D array and write to disk.
            
                    if mode == 'autosaved':
                        channel_files = [f for f in tile_files if f"--C{str(channel).zfill(2)}" in f]
                    elif mode == 'exported':
                        channel_files = [f for f in tile_files if f"_ch{channel}" in f]
                    
                    stacked_images = np.stack([
                        tifffile.imread(os.path.join(input_dir, f)) for f in channel_files
                    ])
                    dw_input = os.path.join(dw_tmp_dir, f'Base_{base}_s{tile}_C0{channel}.tif')
                    tifffile.imwrite(dw_input, stacked_images)
                   
                    # ----- Step 7b: Deconvolve stacked z-planes with Deconwolf -----
                    # Use Deconwolf with corresponding PSF and parameters to perform deconvolution.
            
                    deconvolve_image(
                        input_image=dw_input,
                        psf_image=psf_dict[channel],
                        output_image=dw_output,
                        iterations=20,
                        tilesize=chunk_size
                    )
                    
                    # ----- Step 8: Post-process output (mip or full stack) -----
                    # Apply MIP if enabled, or keep full deconvolved stack.

                    if mip:
                        deconvolved_img = tifffile.imread(dw_output)
                        
                        mipped_img = np.max(deconvolved_img, axis=0).astype('uint16')
                        tifffile.imwrite(
                            os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif'),
                            mipped_img)

                        print(f"Mipped images saved in directory: {base_directory}")
                        
                        shutil.rmtree(dw_output_dir)
                        print(f"Deleted directory: {dw_output_dir}")
                
                    else:
                        print(f"Stacked files saved in directory: {dw_output_dir}")
                
                                            
                    tile_channel_end = time.time()
                    print(f"\033[1;37m[Timing] Full deconvolution cycle for Tile {tile}, Channel {channel} took {tile_channel_end - tile_channel_start:.2f} seconds\033[0m")


                # ----- Step 9: Clean up temporary stacked images -----
                # Delete temporary stacked images for this tile.

                if os.path.exists(dw_tmp_dir):
                    shutil.rmtree(dw_tmp_dir)
                    print(f"Deleted directory: {dw_tmp_dir}")

    # ----- Step 10: Final reporting -----
    # Report total script execution time.
    
    script_end_time = time.time()
    print(f"\033[96m[Total Runtime] Full deconvolution pipeline took {(script_end_time - script_start_time)/60:.2f} minutes\033[0m")
    
    return None


# -------------------------------------------------------------------------------------
# LEICA LIF
# -------------------------------------------------------------------------------------

def deconvolve_lif(input_dir, output_dir_prefix, cycle=None, PSF_metadata=None, chunk_size=None,  tile_size_x=2048, tile_size_y=2048, deconvolution_method='redlionfish', mip=True):
    
    """
    Process the images from the given directories.

    Parameters:
    - input_dir: List of directories containing the images to process.
    - output_dir_prefix: Prefix for the output directories.
    - cycle: Number of ISS cycle to be processed.
    - deconvolution_method: 'redlionfish' (gpu) or 'deconwolf' (cpu)
    - image_dimensions: Dimensions of the images (default: [2048, 2048]).
    - PSF_metadata: Metadata for Point Spread Function (PSF) generation.
    - chunk_size [x,y]: Size of chunks for processing. If None, the entire image is processed.
                  Small GPUs will require chunking. Enable if you run out of gRAM
    - mip: Boolean to decide whether to apply maximum intensity projection. Default is True. 
           If mip=false the stack is deconvolved but saved as an image stack without projecting it
    - mode='autosaved', ='exported' if exported via the export function in LasX

    Returns:
    None. Processed images are saved in the output directories.
    """

    # ----- Step 1: Initial validation and input preparation -----
    # Ensure PSF metadata is provided. Normalize input directory names.
    # Print initial info banner for user feedback.
    
    # Import the LifFile reader to access image data and metadata from .lif files
    from readlif.reader import LifFile

    script_start_time = time.time()
    
    print("\033[1;96m>> Using Deconvolution method: {} <<\033[0m".format(
        "Deconwolf" if deconvolution_method == "deconwolf" 
        else "RedLionFish" if deconvolution_method == "redlionfish" 
        else "Unknown"))
    
    if PSF_metadata is None:
        raise ValueError("PSF_metadata is required to generate PSF.")
  
    input_dir = input_dir.replace("%20", " ")

    # Ensure the output folder exists
    os.makedirs(output_dir_prefix, exist_ok=True)
    
    # ----- Step 2: Detect number of regions and determine LIF file structure -----
    # Extract relevant LIF files, identify number of regions, and set processing mode based on file structure.
    
    base = cycle
    print(f"\033[1;90mProcessing Cycle {base} \033[0m")
  
    lif_files = [f for f in os.listdir(input_dir) if f.endswith('.lif')]
    num_files = len(lif_files)

    # Determine if LIF files are exported (multiple files) or saved (single file with multiple images)
    if num_files > 1:
        # Each file represents one region (exported mode)
        num_regions = num_files
        mode = 'exported'

    elif num_files == 1:
        # Single LIF file, may contain multiple images/regions (saveas mode)
        filepath = os.path.join(input_dir, lif_files[0])
        file = LifFile(filepath)
        num_regions = len(file.image_list)
        mode = 'saveas'


    # ----- Step 3: Prepare image data into consistent format -----
    images = []
    image_dict_list = []
    
    if mode == 'exported':
        for idx in range(num_regions):
            filepath = os.path.join(input_dir,lif_files[idx])
            file = LifFile(filepath)
            
            image_dict = file.image_list[0]

            image_dict_list.append(image_dict)        # Collect image metadata dictionaries
            images.append(image)                      # Collect image data arrays

    elif mode == 'saveas':
        for idx, image_dict in enumerate(file.image_list):
            image = file.get_image(idx)               # Load image data for each region

            image_dict_list.append(image_dict)        # Collect image metadata dictionaries
            images.append(image)                      # Collect image data arrays
            
    # ----- Step 4: Process each region -----
    for region_index in range(num_regions): 
        print(f"\033[1;90mProcessing Region {region_index + 1}/{num_regions}\033[0m")
        
        # Create output subfolders for each region if multiple regions exist
        if num_regions == 1:
            output_directory = output_dir_prefix
        else:
            output_directory = f"{output_dir_prefix}_R{region_index + 1}"
            
        mipped_directory = os.path.join(output_directory, 'preprocessing', 'mipped')
        os.makedirs(mipped_directory, exist_ok=True)

        base_directory = os.path.join(mipped_directory, f'Base_{base}')
        os.makedirs(base_directory, exist_ok=True)


        # ----- Step 5: Load image data -----
        image_dict = image_dict_list[region_index]
        image_name = image_dict['name']
        
        print(f"Image name {region_index}: {image_name}")
        image_name = image_name.replace('/', '_')

        channels = image_dict['channels']
        dims = image_dict['dims'] 
        size_z = dims.z
        n_tiles = dims.m
        mosaic = image_dict.get('mosaic_position', None) # Extract mosaic tile positions if available

             
        # ----- Step 6: Generate XML metadata for tile positions -----
        os.makedirs(os.path.join(base_directory, 'MetaData'), exist_ok=True)
     
        data = ET.Element("Data")
        image_elem = ET.SubElement(data, "Image", TextDescription="")
        attachment = ET.SubElement(
            image_elem, 
            "Attachment", 
            Name="TileScanInfo", 
            Application="LAS AF", 
            FlipX="0", 
            FlipY="0", SwapXY="0")

        for x, y, pos_x, pos_y in mosaic:
            ET.SubElement(
                attachment, 
                "Tile", 
                FieldX=str(x), 
                FieldY=str(y),
                PosX=f"{pos_x:.10f}", 
                PosY=f"{pos_y:.10f}")

        tree = ET.ElementTree(data)
        tree.write(
            os.path.join(base_directory, 'MetaData', f"{image_name}.xml"),
            encoding="utf-8", 
            xml_declaration=True)

        # ----- Step 7: Perform deconvolution -----
        # RedLionFish and Deconwolf implementations differ here.
        
        if deconvolution_method == 'redlionfish':

            # ----- Step 7a: Generate PSFs for all channels -----
            print('Calculating the PSF')
            
            psf_dict = {}
            for idx, channel in enumerate(sorted(PSF_metadata['channels'])):
                psf_dict[idx] = fd_psf.GibsonLanni(
                    na=float(PSF_metadata['na']),
                    m=float(PSF_metadata['m']),
                    ni0=float(PSF_metadata['ni0']),
                    res_lateral=float(PSF_metadata['res_lateral']),
                    res_axial=float(PSF_metadata['res_axial']),
                    wavelength=float(PSF_metadata['channels'][channel]['wavelength']),
                    size_x=tile_size_x,
                    size_y=tile_size_y,
                    size_z=size_z
                ).generate()
                
            # ----- Step 7b: Deconvolve each tile and channel -----
            print("Single tile imaging." if n_tiles == 1 else f"Number of tiles: {n_tiles}")
            
            for tile in range(n_tiles):
                for channel in range(channels):
                    print(f"\033[90m[Cycle {base}] Tile {tile}, Channel {channel}...\033[0m")
                    tile_channel_start = time.time()

                    output_file_path = os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif')
                
                    if os.path.exists(output_file_path):
                        print(f"File {output_file_path} already exists. Skipping.")
                        continue
    
                    # Stack Z-planes ------------------
                    z_planes = []
                    for z_frame in image.get_iter_z(m=tile, c=channel): 
                        z_data = np.array(z_frame)
                        z_planes.append(z_data)
                    stacked_images = np.stack(z_planes, axis=0)
                
                    # Run RedLionFish deconvolution -----
                    deconvolved_img = rl.doRLDeconvolutionFromNpArrays(
                        stacked_images, psf_dict[channel], niter=50)
        
                    # Post-process output (MIP or full stack) -----
                    if mip:
                        processed_img = np.max(deconvolved_img, axis=0).astype('uint16')
                    else:
                        processed_img = deconvolved_img.astype('uint16')
        
                    tifffile.imwrite(output_file_path, processed_img)
                    print(f"Mipped images saved in directory: {base_directory}")

                    tile_channel_end = time.time()
                    print(f"\033[1;37m[Timing] Full deconvolution cycle for Tile {tile}, Channel {channel} took {tile_channel_end - tile_channel_start:.2f} seconds\033[0m")

        elif deconvolution_method == 'deconwolf':
        
            # ----- Step 7a: Generate PSFs for all channels -----
            print('Calculating the PSF')
         
            psf_filepath = os.path.join(base_directory, 'PSF')
            os.makedirs(psf_filepath, exist_ok=True)
            
            psf_dict = {}
            for channel, info in PSF_metadata['channels'].items():
                wavelength_nm = float(info['wavelength']) * 1000
                psf_filename = os.path.join(psf_filepath, f"PSF_channel_{channel}.tif")
                generate_psf(
                    psf_output=psf_filename,
                    resxy=PSF_metadata['res_lateral'] * 1000,
                    resz=PSF_metadata['res_axial'] * 1000,
                    wavelength=wavelength_nm,
                    NA=PSF_metadata['na'],
                    ni=PSF_metadata['ni0']
                )
                psf_dict[channel] = psf_filename
    
            # ----- Step 7b: Deconvolve each tile and channel -----
            print("Single tile imaging." if dims.m == 1 else f"Number of tiles: {dims.m}")
            
            for tile in range(n_tiles):
                dw_tmp_dir = os.path.join(base_directory, 'deconwolf tmp')
                os.makedirs(dw_tmp_dir, exist_ok=True)
                
                for channel in range(channels):
                    print(f"\033[90m[Cycle {base}] Tile {tile}, Channel {channel}...\033[0m")
                    tile_channel_start = time.time()
    
                    dw_output_dir = os.path.join(base_directory, 'stacked')
                    os.makedirs(dw_output_dir, exist_ok=True)
    
                    dw_output = os.path.join(dw_output_dir, f'Base_{base}_s{tile}_C0{channel}.tif')
                    output_file_path = os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif')
    
                    if os.path.exists(output_file_path):
                        print(f"File {output_file_path} already exists. Skipping.")
                        continue
    
                    # Stack Z-planes ------------------
                    z_planes = []
                    for z_frame in image.get_iter_z(m=tile, c=channel):
                        z_data = np.array(z_frame)
                        z_planes.append(z_data)
                    stacked_images = np.stack(z_planes, axis=0)
    
                    dw_input = os.path.join(dw_tmp_dir, f'Base_{base}_s{tile}_C0{channel}.tif')
                    tifffile.imwrite(dw_input, stacked_images)
    
                    # Run Deconwolf ------------------
                    deconvolve_image(
                        input_image=dw_input,
                        psf_image=psf_dict[str(channel)],
                        output_image=dw_output,
                        iterations=20,
                        tilesize=chunk_size
                    )
    
                    # MIP or save full stack -----
                    if mip:
                        deconvolved_img = tifffile.imread(dw_output)
                        mipped_img = np.max(deconvolved_img, axis=0).astype('uint16')
                        tifffile.imwrite(
                            os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif'),
                            mipped_img)
                
                        print(f"Mipped images saved in directory: {base_directory}")
                        shutil.rmtree(dw_output_dir)
                        print(f"Deleted directory: {dw_output_dir}")
                
                    else:
                        print(f"Stacked files saved in directory: {dw_output_dir}")

                    tile_channel_end = time.time()
                    print(f"\033[1;37m[Timing] Full deconvolution cycle for Tile {tile}, Channel {channel} took {tile_channel_end - tile_channel_start:.2f} seconds\033[0m")
                

            # ----- Step 7c: Clean up temporary files -----
            if os.path.exists(dw_tmp_dir):
                shutil.rmtree(dw_tmp_dir)
                print(f"Deleted directory: {dw_tmp_dir}")

    # ----- Step 8: Final reporting -----
    # Report total script execution time.
    
    script_end_time = time.time()
    print(f"\033[96m[Total Runtime] Full deconvolution pipeline took {(script_end_time - script_start_time)/60:.2f} minutes\033[0m")
    
    return None


'''
This functions has been developed around a dataset that is not representative of the typical nd2 format
Tiles should be in the 'm' loop while in this case they are in the 'p' loop which I think it is for positions of
single FOVs.
def deconvolve_nd2 (input_file, outpath, mip=True, PSF_metadata=None, cycle=0):
    """
    Process nd2 files, deconvolve and apply maximum intensity projection (if specified), 
    and create an associated XML with metadata.
    
    Parameters:
    - input_file: Path to the input nd2 file.
    - outpath: Directory where the processed images and XML will be saved.
    - mip: Boolean to decide whether to apply maximum intensity projection. Default is True.
    - cycle: Int to specify the cycle number. Default is 0.
    
    Returns:
    - A string indicating that processing is complete.
    """
    
    # import packages 
    import os
    import pandas as pd
    import re

    import xml.etree.ElementTree as ET
    import nd2
    from xml.dom import minidom
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
    import tifffile
    
    # Create the output directory if it doesn't exist.
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print ('Reading .ND2 file...')
    # Load the nd2 into array and retrieve its dimensions.
    big_file = nd2.imread(input_file)
     
    chsize = big_file.shape[2]
    msize=big_file.shape[0]
    z_size=big_file.shape[1]
    ndfile = nd2.ND2File(input_file)
    tile_size_x = big_file.shape[3]
    tile_size_y = big_file.shape[4]

    # Check if mip is True and cycle is not zero.
    if mip and cycle != 0:
        print ('Extracting the metadata')
        # Initialize placeholders for metadata.
        Bxcoord = []
        Bycoord = []
        Btile_index = []
        filenamesxml = []
        Bchindex = []
        data_str=str(ndfile.experiment)
        ndfile.close()
        split_data = data_str.split('points=', 1)
        positions_str = split_data[1]

        # Remove the '])' from the end of the positions string
        positions_str = positions_str[:-2]

        # Split the positions string at each 'Position('
        positions_list = positions_str.split('Position(')[1:]

        # Initialize an empty list to store the Position() lines
        positions_lines = []

        # Iterate through the positions list and extract each line
        for position in positions_list:
            # Remove the trailing ')' from the line
            position_line = position.split(')')[0]
            # Append the position line to the positions_lines list
            positions_lines.append('Position(' + position_line + ')')

        # Initialize an empty list to store the extracted coordinates
        coordinates = []

        # Iterate through each line of Position() and extract x and y coordinates
        for line in positions_lines:
            # Use regular expressions to extract x and y coordinates
            match = re.search(r'x=([-+]?\d*\.\d+|\d+), y=([-+]?\d*\.\d+|\d+)', line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                coordinates.append({'x': x, 'y': y})

        # Create a DataFrame from the extracted coordinates
        df_coord = pd.DataFrame(coordinates)
        
        # Loop through each mosaic tile and each channel.
        for m in tqdm(range(0, msize)):
            print ('generating the PSF')
            psf_dict = {}
            for idx, channel in enumerate(sorted(PSF_metadata['channels'])):
                psf_dict[idx] = fd_psf.GibsonLanni(
                    na=float(PSF_metadata['na']),
                    m=float(PSF_metadata['m']),
                    ni0=float(PSF_metadata['ni0']),
                    res_lateral=float(PSF_metadata['res_lateral']),
                    res_axial=float(PSF_metadata['res_axial']),
                    wavelength=float(PSF_metadata['channels'][channel]['wavelength']),                
                    size_x=tile_size_x,
                    size_y=tile_size_y,
                    size_z=z_size  # Use the Z dimension from the CZI file
                ).generate()
            for ch in range (0, chsize):
                print ('Deconvolving channel '+str(ch))
                # Get metadata and image data for the current tile and channel.
                #meta = czi.get_mosaic_tile_bounding_box(M=m, Z=0, C=ch)
                z_stack=(big_file[m, :, ch, :, :])
                deconvolved = rl.doRLDeconvolutionFromNpArrays(z_stack, psf_dict[ch], niter=50)
                IM_MAX = np.max(deconvolved, axis=0)
                
                # Construct filename for the processed image.
                n = str(0)+str(m+1) if m < 9 else str(m+1)
                filename = 'Base_' + str(cycle) + '_c' + str(ch+1) + 'm' + str(n) + '_ORG.tif'
                
                # Save the processed image.
                print ('Saving projected image')
                tifffile.imwrite(os.path.join(outpath, filename), IM_MAX.astype('uint16'))
                
                # Append metadata to the placeholders.
                Bchindex.append(ch)
                Bxcoord.append(df_coord.loc[m][0])
                Bycoord.append(df_coord.loc[m][1])
                Btile_index.append(m)
                filenamesxml.append(filename)

        # Adjust the XY coordinates to be relative.
        nBxcord = [x - min(Bxcoord) for x in Bxcoord]
        nBycord = [y - min(Bycoord) for y in Bycoord]
        
        # Create a DataFrame to organize the collected metadata.
        metadatalist = pd.DataFrame({
            'Btile_index': Btile_index, 
            'Bxcoord': nBxcord, 
            'Bycoord': nBycord, 
            'filenamesxml': filenamesxml,
            'channelindex': Bchindex
        })
        
        metadatalist = metadatalist.sort_values(by=['channelindex','Btile_index'])
        metadatalist = metadatalist.reset_index(drop=True)

        # Initialize the XML document structure.
        export_doc = ET.Element('ExportDocument')
        
        # Populate the XML document with metadata.
        for index, row in metadatalist.iterrows():
            image_elem = ET.SubElement(export_doc, 'Image')
            filename_elem = ET.SubElement(image_elem, 'Filename')
            filename_elem.text = row['filenamesxml']
            
            bounds_elem = ET.SubElement(image_elem, 'Bounds')
            bounds_elem.set('StartX', str(row['Bxcoord']))
            bounds_elem.set('SizeX', str(big_file.shape[3]))
            bounds_elem.set('StartY', str(row['Bycoord']))
            bounds_elem.set('SizeY', str(big_file.shape[4]))
            bounds_elem.set('StartZ', '0')
            bounds_elem.set('StartC', '0')
            bounds_elem.set('StartM', str(row['Btile_index']))
            
            zoom_elem = ET.SubElement(image_elem, 'Zoom')
            zoom_elem.text = '1'

        
        # Save the constructed XML document to a file.
        xml_str = ET.tostring(export_doc)
        with open(outpath + 'Base_' + str(cycle) + '_info.xml', 'wb') as f:
            f.write(xml_str)

    return "Processing complete."
'''