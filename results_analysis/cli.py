import argparse
from pathlib import Path



def parse_compile():
    '''
        Parse the expected command-line arguments for compile_seasonal_water.py
    '''
    parser = argparse.ArgumentParser(description='Compiles seasonal surface water extents from different datasets.')

    # parser.add_help()

    parser.add_argument('-d', '--dataset', type=str,
                        default='ps',
                        help='Input the dataset of interest.')
    
    # parser.add_argument('-s', '--seasonal', type=bool,
    #                     default=True,
    #                     help='Set to True for seasonal classification images.\nSet to False for annual classification images.')

    # parser.add_argument('-o', '--out_dir', type=str,
    #                     default='../open_source_training/',
    #                     help='Output directory for where files will be saved.')
    # parser.add_argument('-p', '--path',
    #                     default=DOWNLOADED_IMG_PATH,
    #                     help='Path where the flood training datasets have been downloaded')

    return(parser.parse_args())

