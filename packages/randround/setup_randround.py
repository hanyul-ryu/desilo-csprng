
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name='randround',
        ext_modules=[
            CUDAExtension('randround_cuda', [
                'randround.cpp',
                'randround_cuda_kernel.cu',
                ])
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
