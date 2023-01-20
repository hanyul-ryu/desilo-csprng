
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name='randint',
        ext_modules=[
            CUDAExtension('randint_cuda', [
                'randint.cpp',
                'randint_cuda_kernel.cu',
                ])
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
