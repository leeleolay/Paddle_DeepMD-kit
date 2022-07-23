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
```

- compile_paddle.sh  
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle  
git reset --hard eca6638c599591c69fe40aa196f5fd42db7efbe2  
rm -rf build && mkdir build && cd build  
cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DWITH_GPU=OFF -DWITH_AVX=ON -DON_INFER=ON -DCMAKE_BUILD_TYPE=Release  
make -j 32  
make -j 32 inference_lib_dist  
```
- compile_deepmd.sh  
```
git clone https://github.com/X4Science/paddle-deepmd.git
rm -rf /home/deepmdroot/ && mkdir /home/deepmdroot && deepmd_root=/home/deepmdroot
cd /home/paddle-deepmd/source && rm -rf build && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$deepmd_root -DPADDLE_ROOT=$paddle_root -DUSE_CUDA_TOOLKIT=FALSE -DFLOAT_PREC=low ..
make -j 4 && make install
make lammps
```
- compile_lammps.sh  
```
#apt install libc-dev
cd /home
wget https://github.com/lammps/lammps/archive/stable_29Oct2020.tar.gz
rm -rf lammps-stable_29Oct2020/
tar -xzvf stable_29Oct2020.tar.gz
cd lammps-stable_29Oct2020/src/
cp -r /home/deepmd-kit/source/build/USER-DEEPMD .
make yes-kspace yes-user-deepmd
#make serial -j 20
make mpi -j 20
```

# 4.Performance
- The performance of inference based on the LAMMPS with PaddlePaddle framework，comparing with TensorFlow framework, about single core and multi-threads
![截屏2022-05-25 23 08 11](https://user-images.githubusercontent.com/50223303/170295703-32e18058-aff9-4368-93cd-38a1ed787e8a.png)
- single thread performance
```
TF_INTRA_OP_PARALLELISM_THREADS=8 TF_INTER_OP_PARALLELISM_THREADS=1 numactl -c 0 -m 0 lmp_serial -in in.lammps
```
- multithreads performance
```
OMP_NUM_THREADS=1 TF_INTRA_OP_PARALLELISM_THREADS=1 TF_INTER_OP_PARALLELISM_THREADS=1  mpirun --allow-run-as-root -np 4 lmp_mpi -in in.lammps
```  
# 5.Future Plans
- fix training precision
- support Gromacs
- support more descriptor and model
- Support GPU

# 6.Cooperation
Welcome to join us to develop this program together.  
Please contact us from [X4Science](https://github.com/X4Science) [PaddlePaddle](https://www.paddlepaddle.org.cn) [PPSIG](https://www.paddlepaddle.org.cn/sig) [PaddleAIforScience](https://www.paddlepaddle.org.cn/science) [PaddleScience](https://github.com/PaddlePaddle/PaddleScience).
