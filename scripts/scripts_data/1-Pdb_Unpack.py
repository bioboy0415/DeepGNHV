import os
import argparse
import gzip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def un_gz(file_name):
    file_path_input  = compressed_dir/file_name
    
    file_name_valid  = '-'.join(file_name.split('-')[1:3])
    file_path_output = af2_dir/f'{file_name_valid}.pdb'
    
    g_file    = gzip.GzipFile(file_path_input)
    open(file_path_output, 'wb+').write(g_file.read())
    g_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--compressed_dir', type = str, help = 'Path for compressed pdb files directory.')
    parser.add_argument('--af2_dir',        type = str, help = 'Path for unpacked pdb files directory.')
    
    args = parser.parse_args()
    
    compressed_dir = Path(args.compressed_dir)
    af2_dir        = Path(args.af2_dir)
    af2_dir.mkdir(exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        req = [executor.submit(un_gz, item) for item in os.listdir(compressed_dir)]