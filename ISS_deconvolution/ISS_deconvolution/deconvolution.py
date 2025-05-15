from os import listdir
from os.path import isfile, join
from xml.dom import minidom
import pandas as pd
import numpy as np
import re
import shutil
import os
import tifffile
import RedLionfishDeconv as rl
import matplotlib.pyplot as plt
#from flowdec.nb import utils as nbutils 
#from flowdec import psf as fd_psf
#from flowdec import data as fd_data
from scipy import ndimage
import dask
import dask.array as da
#import tensorflow as tf
#from flowdec.restoration import RichardsonLucyDeconvolver
from skimage import io
from pathlib import Path
import operator
#from flowdec import data as fd_data
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
from aicspylibczi import CziFile
import ISS_deconvolution.psf as fd_psf
from readlif.reader import LifFile
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

def pad_numbers(numbers):
    """
    Pads single-digit numbers with zeros based on the length of the list.

    Args:
      numbers: A list of numbers.

    Returns:
      A list of numbers with appropriate zero padding.
    """
    length = len(numbers)

    if length < 10:
        return numbers  # No padding needed

    if 10 <= length < 100:
        padded_numbers = [f"{num:02}" for num in numbers]
        return padded_numbers

    if length >= 100:
        padded_numbers = [f"{num:03}" for num in numbers]
        return padded_numbers






def deconvolve_czi(input_file, outpath, image_dimensions=[2048, 2048], PSF_metadata=None,  mip=True, cycle=0, tile_size_x=2048, tile_size_y=2048):

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
	
    def customcopy(src, dst):
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        shutil.copyfile(src, dst)

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
           

                
                deconvolved = rl.doRLDeconvolutionFromNpArrays(img, psf_dict[idx], niter=50)
                if chsize != len(PSF_metadata['channels']):
                    raise ValueError("Mismatch between CZI file channels and PSF_metadata channels.")
                #print(deconvolved.data.shape)
                # Check if mip (max intensity projection) is enabled
                if mip:
                    processed_img = np.max(deconvolved, axis=0).astype('uint16')
                else:
                    processed_img = deconvolved.astype('uint16')
            
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
    




def deconvolve_leica(input_dirs, 
                     output_dir_prefix, 
                     image_dimensions=[2048, 2048], 
                     PSF_metadata=None, 
                     chunk_size=None, 
                     mip=True,
                    mode=None):
    """
    Process the images from the given directories.

    Parameters:
    - input_dirs: List of directories containing the images to process.
    - output_dir_prefix: Prefix for the output directories.
    - image_dimensions: Dimensions of the images (default: [2048, 2048]).
    - PSF_metadata: Metadata for Point Spread Function (PSF) generation.
    - chunk_size [x,y]: Size of chunks for processing. If None, the entire image is processed.
                  Small GPUs will require chunking. Enable if you run out of gRAM
    - mip: Boolean to decide whether to apply maximum intensity projection. Default is True. 
           If mip=false the stack is deconvolved but saved as an image stack without projecting it
    - mode=None if autosaved, ='exported' if exported via the export function in LasX

    Returns:
    None. Processed images are saved in the output directories.
    """
    def cropND(img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(lambda a, da: a+da, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]
    
    if PSF_metadata is None:
        raise ValueError("PSF_metadata is required to generate PSF.")

    def custom_copy(src, dest):
        """Custom function to copy a file to a destination."""
        if os.path.isdir(dest):
            dest = os.path.join(dest, os.path.basename(src))
        shutil.copyfile(src, dest)
    
    if mode==None:

        # Preprocess directory names (replace placeholders)
        processed_dirs = [dir_name.replace("%20", " ") for dir_name in input_dirs]  # Replace placeholder for spaces

        # Loop through directories to process images
        for dir_index, dir_path in enumerate(processed_dirs):
            tif_files = [f for f in os.listdir(dir_path) if '.tif' in f and 'dw' not in f and '.txt' not in f]
            
            # Extract unique region identifiers from filenames
            regions = pd.DataFrame(tif_files)[0].str.split('--', expand=True)[0].unique()

        
            # Process each region
            for region in regions:
                print('Counting the regions and organizing the files')
                filtered_tifs = [f for f in tif_files if region in f]
                base_num = str(dir_index + 1)
                
                # Extract unique tile identifiers
                tiles_df = pd.DataFrame(filtered_tifs)[0].str.split('--', expand=True)[1]
                tiles_df = tiles_df.str.extract(r'(\d+)')[0].sort_values().unique()

                # Determine output directory based on number of regions
                if len(regions) == 1:
                    output_directory = output_dir_prefix
                    mipped_directory = os.path.join(output_directory, 'preprocessing', 'mipped')
                else:
                    output_directory = f"{output_dir_prefix}_R{region.split('Region')[1].split('_')[0]}"
                    mipped_directory = os.path.join(output_directory, 'preprocessing', 'mipped')

                # Ensure the output directory exists
                os.makedirs(mipped_directory, exist_ok=True)
                
                # Process each cycle
                for base in sorted(base_num):
                    base_directory = os.path.join(mipped_directory, f'Base_{base}')
                    os.makedirs(base_directory, exist_ok=True)
                    
                    # Copy metadata if it exists
                    print ('Extracting metadata')
                    metadata_dir = os.path.join(dir_path, 'Metadata')
                    metadata_files = [f for f in os.listdir(metadata_dir) if region in f]
                    if metadata_files:
                        os.makedirs(os.path.join(base_directory, 'MetaData'), exist_ok=True)
                        custom_copy(os.path.join(metadata_dir, metadata_files[0]), os.path.join(base_directory, 'MetaData'))
                    
                    # Extracts the first tile to calculate its Z depth
                    print ('Calculating the PSF')
                    sample_tile=[f for f in filtered_tifs if f"--Stage00--" in f]
                    size_z = int(len(sample_tile) / len(PSF_metadata['channels']))
                    # Generate PSFs for each channel outside the tile loop
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
                    # Process each tile within the base
                    print ('Deconvolving Cycle: '+ base)
                    for tile in tqdm(sorted(tiles_df, key=int)):
                    #for tile in sorted(tiles_df, key=int):
                        tile_files = [f for f in filtered_tifs if f"--Stage{tile}--" in f]
                        for channel in sorted(PSF_metadata['channels']):
                            output_file_path = os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif')

                            if os.path.exists(output_file_path):
                                print(f"File {output_file_path} already exists. Skipping this tile for channel {channel}.")
                                continue
                            
                            channel_files = [f for f in tile_files if f"--C0{channel}" in f]
                            stacked_images = np.stack([tifffile.imread(os.path.join(dir_path, f)) for f in channel_files])

                            deconvolved = rl.doRLDeconvolutionFromNpArrays(stacked_images, psf_dict[channel], niter=50)
                            
                            if mip:
                                processed_img = np.max(deconvolved, axis=0).astype('uint16')
                            else:
                                processed_img = np.asarray(deconvolved).astype('uint16')

                            tifffile.imwrite(os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif'), processed_img)
                            
                            
    if mode=='exported':
        print ('Deconvolving Leica files from export mode')
        #Preprocess directory names (replace placeholders)
        processed_dirs = [dir_name.replace("%20", " ") for dir_name in input_dirs]  # Replace placeholder for spaces

        # Loop through directories to process images
        

        for dir_index, dir_path in enumerate(processed_dirs):
            tif_files = [f for f in os.listdir(dir_path) if '.tif' in f and 'dw' not in f and '.txt' not in f]
            split_underscore = pd.DataFrame(tif_files)[0].str.split('_', expand=True)
            regions = list(split_underscore[0].unique())

        
            # Process each region
            for region in regions:
                #print (region)
                print('Counting the regions and organizing the files')
                filtered_tifs = [f for f in tif_files if region in f]
                base_num = str(dir_index + 1)
                
                # Extract unique tile identifiers
                split_underscore = pd.Series(filtered_tifs).str.split('_', expand=True)

                #tiles = sorted(split_underscore.iloc[:, -3].unique())
                #tiles = pd.Series(filtered_tifs).str.extract(r'_s(\d+)_')[0].dropna().astype(int)
                #tiles = sorted(tiles.unique()) 
                #print (tiles)
                #tiles = pad_numbers(tiles)
                
                tiles = pd.Series(filtered_tifs).str.extract(r'_s(\d+)_')[0].dropna() # Import as strings
                #print('tiles: ',tiles)
                tiles = sorted(tiles.unique(),key=int) # Sort as numbers, but keep as strings
                print ('sorted tiles: ',tiles)
		    
                #tiles_df = pd.DataFrame(tiles)
                #tiles_df['indexNumber'] = [int(tile.split('s')[-1]) for tile in tiles_df[0]]
                #tiles_df.sort_values(by=['indexNumber'], ascending=True, inplace=True)
                #tiles_df.drop(labels='indexNumber', axis=1, inplace=True)
                #tiles = list(tiles_df[0])

                # Determine output directory based on number of regions
                if len(regions) == 1:
                    output_directory = output_dir_prefix
                    mipped_directory = os.path.join(output_directory, 'preprocessing', 'mipped')
                else:
                    output_directory = f"{output_dir_prefix}_R{region.split('Region')[1].split('_')[0]}"
                    mipped_directory = os.path.join(output_directory, 'preprocessing', 'mipped')

                # Ensure the output directory exists
                os.makedirs(mipped_directory, exist_ok=True)
                #print (mipped_directory)
                
                # Process each cycle
                for base in sorted(base_num):
                    base_directory = os.path.join(mipped_directory, f'Base_{base}')
                    os.makedirs(base_directory, exist_ok=True)
                    #print(base_directory)
                    # Copy metadata if it exists
                    print ('Extracting metadata')
                    metadata_dir = os.path.join(dir_path, 'Metadata')
                    metadata_files = [f for f in os.listdir(metadata_dir) if region in f]
                    if metadata_files:
                        os.makedirs(os.path.join(base_directory, 'MetaData'), exist_ok=True)
                        custom_copy(os.path.join(metadata_dir, metadata_files[0]), os.path.join(base_directory, 'MetaData'))
                    # Extracts the first tile to calculate its Z depth
                    print ('Calculating the PSF')
                    #print (filtered_tifs)

                    if len(tiles)>100:
                        sample_tile=[f for f in filtered_tifs if f"_s000_" in f]
                    elif len(tiles)>10 and len(tiles)<100:
                        sample_tile=[f for f in filtered_tifs if f"_s00_" in f]
                    else:
                        sample_tile=[f for f in filtered_tifs if f"_s0_" in f]
                    #sample_tile=filtered_tifs[0]
                    #print (sample_tile)
                    
                    
                    size_z = int(len(sample_tile) / len(PSF_metadata['channels']))
                    print (size_z)
                    # Generate PSFs for each channel outside the tile loop
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
                    
                    # Process each tile within the base
                    print ('Deconvolving Cycle: '+ base)
                    for tile in tqdm(sorted(tiles, key=int)):
                        print (tile)
                   # for tile in sorted(tiles_df, key=int):
                        tile_files = [f for f in filtered_tifs if f"_s{tile}" in f]
                        print (tile_files)
                        for channel in sorted(PSF_metadata['channels']):
                            print ("processing channel: "+channel)
                            output_file_path = os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif')

                            if os.path.exists(output_file_path):
                                print(f"File {output_file_path} already exists. Skipping this tile for channel {channel}.")
                                continue
                            
                            channel_files = [f for f in tile_files if f"_ch{channel}" in f]
                            #print (channel)
                            stacked_images = np.stack([tifffile.imread(os.path.join(dir_path, f)) for f in channel_files])
                                
                            deconvolved = rl.doRLDeconvolutionFromNpArrays(stacked_images, psf_dict[channel], niter=50)

                            
                            if mip:
                                processed_img = np.max(deconvolved, axis=0).astype('uint16')
                            else:
                                processed_img = np.asarray(deconvolved).astype('uint16')

                            tifffile.imwrite(os.path.join(base_directory, f'Base_{base}_s{tile}_C0{channel}.tif'), processed_img)
                            
    return None

def max_deconvolve_lif_stack(image, m, c, psf_dict):
    # Initialize a list to hold all Z-plane data
    z_planes = []

    # Iterate over all Z-planes for the given timepoint and channel
    for z_frame in image.get_iter_z(m=m, c=c):
        # Convert the Pillow Image to a NumPy array
        z_data = np.array(z_frame)
        #print (z_data.shape)
        z_planes.append(z_data)

    # Stack all Z-planes along a new axis
    z_stack = np.stack(z_planes, axis=0)
    
    deconvolved = rl.doRLDeconvolutionFromNpArrays(z_stack, psf_dict[c],method='gpu', niter=50)
    #print (z_stack.shape)

    # Perform maximum intensity projection along the Z-axis (axis=0)
    max_projection = np.max(deconvolved, axis=0)

    return max_projection








def lif_deconvolution(lif_path, output_folder, PSF_metadata=None, cycle=None, tile_size_x=2048, tile_size_y=2048):
    from readlif.reader import LifFile
    file = LifFile(lif_path)
    
    os.makedirs(output_folder, exist_ok=True)
    if len(file.image_list) > 1:
        for index, image_dict in enumerate(file.image_list):
            image_name = image_dict['name']
            print (image_name)
            mosaic=image_dict.get('mosaic_position', None)
            #print (mosaic)
            
            # Build XML structure
            data = ET.Element("Data")
            image = ET.SubElement(data, "Image", TextDescription="")
            attachment = ET.SubElement(image, "Attachment", Name="TileScanInfo", Application="LAS AF", FlipX="0", FlipY="0", SwapXY="0")

            for x, y, pos_x, pos_y in mosaic:
                ET.SubElement(attachment, "Tile", FieldX=str(x), FieldY=str(y),
                              PosX=f"{pos_x:.10f}", PosY=f"{pos_y:.10f}")

            # Create tree and write to file
            tree = ET.ElementTree(data)

            regionID=index+1
            output_region=f'_R{regionID}'
            output_subfolder=os.path.join(output_folder, output_region)
            mipped_subfolder = f"{output_subfolder}/preprocessing/mipped/Base_{cycle}"
            os.makedirs(mipped_subfolder, exist_ok=True)
            os.makedirs(mipped_subfolder+'/MetaData', exist_ok=True)
            print(f"Extracting metadata for: {image_name}")
            image_name = image_name.replace('/', '_')
            tree.write(f"{mipped_subfolder}/MetaData/{image_name}.xml", encoding="utf-8", xml_declaration=True)
            print(f"Processing Image {index}: {image_name}")
            image = file.get_image(index)
            channels = image_dict['channels']
            dims = image_dict['dims']
            z_size=dims.z


            if dims.m == 1:
                print("Single tile imaging.")
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
                for c in range(channels):  # Loop through each channel
                    max_projected = max_deconvolve_lif_stack(image, m, c, psf_dict)
                    # Clean filename
                    clean_name = f"Base_{cycle}"
                    filename = f"{clean_name}_s00_C0{c}.tif"
                    output_path = os.path.join(mipped_subfolder, filename)

                    tifffile.imwrite(output_path, max_projected.astype(np.uint16))
                    print(f"Saved: {output_path}")

            else:
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
                for m in range(dims.m):  # Loop through each tile
                    for c in range(channels):  # Loop through each channel
                        max_projected = max_deconvolve_lif_stack(image, m, c, psf_dict)
                        # Clean filename
                        clean_name = f"Base_{cycle}"
                        filename = f"{clean_name}_s{m:02d}_C0{c}.tif"
                        output_path = os.path.join(mipped_subfolder, filename)

                        tifffile.imwrite(output_path, max_projected.astype(np.uint16))
                        print(f"Saved: {output_path}")
    else:
        mipped_subfolder = f"{output_folder}/preprocessing/mipped/Base_{cycle}"
        os.makedirs(mipped_subfolder, exist_ok=True)
        os.makedirs(mipped_subfolder+'/MetaData', exist_ok=True)

        for index, image_dict in enumerate(file.image_list):
            image_name = image_dict['name']
            mosaic=image_dict.get('mosaic_position', None)
            #print (mosaic)
            
            # Build XML structure
            data = ET.Element("Data")
            image = ET.SubElement(data, "Image", TextDescription="")
            attachment = ET.SubElement(image, "Attachment", Name="TileScanInfo", Application="LAS AF", FlipX="0", FlipY="0", SwapXY="0")

            for x, y, pos_x, pos_y in mosaic:
                ET.SubElement(attachment, "Tile", FieldX=str(x), FieldY=str(y),
                              PosX=f"{pos_x:.10f}", PosY=f"{pos_y:.10f}")

            # Create tree and write to file
            tree = ET.ElementTree(data)
            print(f"Extracting metadata for: {image_name}")
            tree.write(f"{mipped_subfolder}/MetaData/{image_name}.xml", encoding="utf-8", xml_declaration=True)
            
            print(f"Processing Image {index}: {image_name}")
            image = file.get_image(index)
            channels = image_dict['channels']
            dims = image_dict['dims']
            z_size=dims.z

            if dims.m == 1:
                print("Single tile imaging.")
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
                for c in range(channels):  # Loop through each channel
                    max_projected = max_deconvolve_lif_stack(image, m, c, psf_dict)
                    # Clean filename
                    clean_name = f"Base_{cycle}"
                    filename = f"{clean_name}_s00_C0{c}.tif"
                    output_path = os.path.join(mipped_subfolder, filename)

                    tifffile.imwrite(output_path, max_projected.astype(np.uint16))
                    print(f"Saved: {output_path}")

            else:
                
                for m in range(dims.m):  # Loop through each tile
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
               # Loop through each tile
                    for c in range(channels):  # Loop through each channel
                        max_projected = max_deconvolve_lif_stack(image, m, c, psf_dict)
                        # Clean filename
                        clean_name = f"Base_{cycle}"
                        filename = f"{clean_name}_s{m:02d}_C0{c}.tif"
                        output_path = os.path.join(mipped_subfolder, filename)

                        tifffile.imwrite(output_path, max_projected.astype(np.uint16))
                        print(f"Saved: {output_path}")



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
