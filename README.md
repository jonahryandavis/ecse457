# ecse457

# Dependencies

sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install python-pip
sudo pip install opencv-python
sudo pip install objdict
sudo pip install matplotlib
sudo pip install numpy
sudo pip install scipy

# Using tr32json.py

This program reads the JSON files that are output by TR3 and concatenates them into a more useful format for the rest of the intelligent scissors process.

Options:
    --d <directory_name>: The name of the directory where the script should find all the JSON files
    --o <out_filename>: The filename of the output concatenated JSON
    --xy_calc <float_val>: This is the value of XY_CALC as stored in the TR3 file. MUST BE ACCURATE
    --z_calc <float_val>: This is the value of Z_CALC as stored in the TR3 file. MUST BE ACCURATE

# Using json2img.py

This program reads the JSON file output by the previous script, and converts it to an "edge map" image.

Options:
    --j <json_filename>: The JSON file to read
    --d <out_directory>: The output directory to create the images in
    --o <output_file_prefix>: The file prefix for each of the output image files
    --x_scale <int>: The width of the image output in px, should match the width of the input image set. MUST BE ACCURATE
    --y_scale <int>: The height of the image output in px, should match the height of the input image set. MUST BE ACCURATE

# Using main.py

This program runs the livewire GUI to segment a new image

1. Open the image you wish to segment.
2. Left mouse to place a new anchor/seed. Right mouse to complete the segment. Escape to save the segmentation.
3. Choose a name for the image segmentation and save the file
    
