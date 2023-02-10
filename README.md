# 1.Introduction
This repo is based on the PaddlePaddle deep learning framework including training and inference parts，DeepMD-kit package，LAMMPS software. The target is, basing on PaddlePaddle framework, to accomplish molecular dynamics simulation with deep learning method.
- PaddlePaddle (PArallel Distributed Deep LEarning) is a simple, efficient and extensible deep learning framework.
- DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics (MD).
- LAMMPS is a classical molecular dynamics code with a focus on materials modeling. It's an acronym for Large-scale Atomic/Molecular Massively Parallel Simulator.

# 2.Progress&Features
- Based on Intel CPU, the pipline of training and inference runs smoothly
- Support traditional molecular dynamics software LAMMPS
- Support se_a desciptor model

# 3.Compiling&Building&Installation
- prepare docker and python environment
```
docker pull paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82 
docker run -it --name {name} -v 绝对路径开发目录:绝对路径开发目录 -v /root/.cache:/root/.cache -v /root/.ccache:/root/.ccache {image_id} bash 
rm -f /usr/bin/python3
ln -s /usr/bin/python3.8 /usr/bin/python3
wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz && tar -xf cmake-3.21.0-linux-x86_64.tar.gz
add ~/.bashrc：export PATH=/home/cmake-3.21.0-linux-x86_64/bin:$PATH
```

- compile Paddle
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle  
git reset --hard eca6638c599591c69fe40aa196f5fd42db7efbe2  
rm -rf build && mkdir build && cd build  
cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DWITH_GPU=OFF -DWITH_AVX=ON -DON_INFER=ON -DCMAKE_BUILD_TYPE=Release  
# cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DWITH_GPU=On -DWITH_AVX=ON -DON_INFER=ON -DCUDA_ARCH_NAME=Auto -DCMAKE_BUILD_TYPE=Release
########################################################################
##git checkout e1e0deed64 #virified
##git checkout 23def39672 #need to be verifyied， newer version
##cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_AVX=ON -DWITH_MKLDNN=ON -DON_INFER=ON -DWITH_TESTING=OFF -DWITH_INFERENCE_API_TEST=OFF -DWITH_NCCL=OFF -DWITH_PYTHON=OFF -DWITH_LITE=OFF -DWITH_ONNXRUNTIME=OFF -DWITH_XBYAK=OFF -DWITH_RCCL=OFF -DWITH_CRYPTO=OFF
##make -j`nproc` all
##########################################################################
make -j 32  
make -j 32 inference_lib_dist  
python3 -m pip install python/dist/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl --no-cache-dir
PADDLE_ROOT=/home/Paddle/build/paddle_inference_install_dir(or add in bashrc with export)
```
- compile Paddle_DeepMD-kit --training part 
```
cd /home
git clone https://github.com/X4Science/paddle-deepmd.git
cd /home/paddle-deepmd
python3 -m pip install tensorflow-cpu==2.5.0
# python3 -m pip install tensorflow-gpu==2.5.0
python3 -m pip install scikit-build
python3 setup.py install
find the package name of deepmd-kit in the location of installation and add in bashrc
        export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/**{deepmd_name}**/deepmd/op:$LD_LIBRARY_PATH
        export LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/**{deepmd_name}**/deepmd/op:$LIBRARY_PATH
        export DEEP_MD_PATH=/usr/local/lib/python3.8/dist-packages/**{deepmd_name}**/deepmd/op
source ~/.bashrc
cd deepmd && python3 load_paddle_op.py install
```

- compile Paddle_DeepMD-kit --inference part 
```
rm -rf /home/deepmdroot/ && mkdir /home/deepmdroot && DEEPMD_ROOT=/home/deepmdroot(or add in bashrc with export)
cd /home/paddle-deepmd/source && rm -rf build && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$DEEPMD_ROOT -DPADDLE_ROOT=$PADDLE_ROOT -DUSE_CUDA_TOOLKIT=FALSE -DFLOAT_PREC=low ..
make -j 4 && make install
make lammps
```
- compile LAMMPS
```
cd /home
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7
./configure
make all install
add bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
#apt install libc-dev
cd /home
wget https://github.com/lammps/lammps/archive/stable_29Oct2020.tar.gz
rm -rf lammps-stable_29Oct2020/
tar -xzvf stable_29Oct2020.tar.gz
cd lammps-stable_29Oct2020/src/
cp -r /home/paddle-deepmd/source/build/USER-DEEPMD .
make yes-kspace yes-user-deepmd
#make serial -j 20
add in bashrc by
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/paddle/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/third_party/install/mkldnn/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/third_party/install/mklml/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle/fluid/pybind/:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/deepmd-kit/source/build:$LD_LIBRARY_PATH
source ~/.bashrc
make mpi -j 20
add in bashrc by
        export PATH=/home/lammps-stable_29Oct2020/src:$PATH
```
# 4.Using Guide
example: water
- training
```
cd /paddle_deepmd-kit_PATH/example/water/train/
dp train water_se_a.json
cp ./model.ckpt/model.pd* ../lmp/ -r
cd ../lmp
```
- inference
```
mpirun -np 10 lmp_mpi -in in.lammps
```


# 5.Latency
The performance of inference based on the LAMMPS with PaddlePaddle framework，comparing with TensorFlow framework, about single core and multi-threads

- test commands of Paddle
```
# serial computation(single process of LAMMPS with signle thread of deep learning framework)
OMP_NUM_THREADS=1 lmp_serial -in in.lammps
# parallel computation（multiprocess of LAMMPS with single threads of deep learning framework）
OMP_NUM_THREADS=1 mpirun --allow-run-as-root -np 4 lmp_mpi -in in.lammps
```

- test commands of Tensorflow 
```
# serial computation(single process of LAMMPS with signle thread of deep learning framework)
TF_INTRA_OP_PARALLELISM_THREADS=1 TF_INTER_OP_PARALLELISM_THREADS=1 numactl -c 0 -m 0 lmp_serial -in in.lammps
# parallel computation（multiprocess of LAMMPS with single thread of deep learning framework）
TF_INTRA_OP_PARALLELISM_THREADS=1 TF_INTER_OP_PARALLELISM_THREADS=1  mpirun --allow-run-as-root -np 4 lmp_mpi -in in.lammps
``` 
![图片1](https://user-images.githubusercontent.com/50223303/203763299-871b66d1-d731-4ab0-ab3e-dba6d35b613b.png)
The ordinate represents seconds

The test of multiprocess with multithreads of TF on the Baidu internel machine with Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
```
## No.1:6min53seconds No.2:8min36seconds
TF_INTRA_OP_PARALLELISM_THREADS=1 TF_INTER_OP_PARALLELISM_THREADS=1  mpirun --allow-run-as-root -np 4 lmp_mpi -in in.lammps

## No.1:3min49seconds No.2:3min42seconds
TF_INTRA_OP_PARALLELISM_THREADS=4 TF_INTER_OP_PARALLELISM_THREADS=1  mpirun --allow-run-as-root -np 4 lmp_mpi -in in.lammps

## No.1:5min15seconds  No.2:3min33seconds
TF_INTRA_OP_PARALLELISM_THREADS=8 TF_INTER_OP_PARALLELISM_THREADS=1  mpirun --allow-run-as-root -np 4 lmp_mpi -in in.lammps

### This is shared machine, I am not sure the cores of CPU only are occupied by my task
```

# 6.Architecture
![图片2](/Pics/Paddle_DeemMD-kit.png)
# 7.Future Plans
- add&fix more descriptors using Paddle
- add DeepTensor model and Dipole Polar net using Paddle
- add custom ops used by DeepTensor model and descriptors using Paddle
- fix Gromacs interface
- support GPU trainning
- support cluster inference

# 8.Cooperation
Welcome to join us to develop this program together.  
Please contact us from [X4Science](https://github.com/X4Science) [PaddlePaddle](https://www.paddlepaddle.org.cn) [PPSIG](https://www.paddlepaddle.org.cn/sig) [PaddleAIforScience](https://www.paddlepaddle.org.cn/science) [PaddleScience](https://github.com/PaddlePaddle/PaddleScience).
