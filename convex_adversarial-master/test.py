import setGPU
import torch

if __name__ == "__main__":
    uid = int(subprocess.run(['id', '-u'], stdout=subprocess.PIPE).stdout)
    gid = int(subprocess.run(['id', '-g'], stdout=subprocess.PIPE).stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, help='command to run')
    
    args = parser.parse_args()
    docker_run(args.cmd)