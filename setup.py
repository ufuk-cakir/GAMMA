from setuptools import setup, find_packages

setup(
    name='gamma',
    version='0.1.0.dev0',
    author='Ufuk Çakır',
    author_email='ufukcakir2001@gmail.com',
    description='GAMMA: Galactic Attributes of Mass, Mettallicity and Age',
    url='https://github.com/ufuk-cakir/GAMMA',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
