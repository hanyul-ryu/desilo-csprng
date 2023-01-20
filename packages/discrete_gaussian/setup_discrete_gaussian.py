
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name='discrete_gaussian',
        ext_modules=[
            CUDAExtension('discrete_gaussian_cuda', [
                'discrete_gaussian.cpp',
                'discrete_gaussian_cuda_kernel.cu',
                ])
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
