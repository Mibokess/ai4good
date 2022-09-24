# Ai4Good - Generalising Land Use and Land Cover Mapping Across Geographical Regions for Rapid Infrastructure Planning

### Get segmentation masks and images of Jakarta directly on Euler:

segmentation masks: /cluster/scratch/amilojevic/ai4good/seg_jakarta_builddings an /cluster/scratch/amilojevic/ai4good/seg_jakarta_streets
rgb images:  /cluster/scratch/jehrat/ai4good/splits_jakarta/satellite

### Get generalized segmentation checkpoint files

https://polybox.ethz.ch/index.php/s/ERW3dxsKmLVIrcV

### Run segmenation on jakarta images example

python run_final_segmentation.py --dataset_root /cluster/scratch/jehrat/ai4good/splits_jakarta/satellite --save_root /cluster/scratch/$USER --segnet_checkpoint segnet_streets.ckpt

### If you want to run pipeline from A to Z (we do not recommend)

1. Generate domain X and Y data by running the script get_data_for_domain_adaption.py (takes very long time)
2. If you have image-label pairs for doamin X and domain Y color images, run_train_cycleGAN.py to generate cycleGAN checkpoint file. 
3. If you have image-label pairs for east-coast, i.e. from domain X, run_train_segmentation.py to generate segnet checkpoint file. 
4. If you have both cycleGAN and segmentation checkpoint files you can run_pseudoseg_experiment.py 

! Warning ! This pipeline was not thoroughly tested because we ran out of time for submission

What you can always do, is run our own generated checkpoint files for the resulting segnet of our pseuo-label approach with run_final_segmentation.py with the checkpoint files mentioned above.

### Download Data

The 'mask_functions.ipynb' notebook contains the functions to download masks and satellite images from Open Street Maps respectively Google Earth Engine.
