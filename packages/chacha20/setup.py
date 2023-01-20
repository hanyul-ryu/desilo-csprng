
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name='chacha20',
        ext_modules=[
            CUDAExtension('chacha20_cuda', [
                'chacha20.cpp',
                'chacha20_cuda_kernel.cu',
                ])
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
