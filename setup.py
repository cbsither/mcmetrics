from setuptools import setup, find_packages

written_descsription = 'Monte Carlo Metrics for evaluating machine ' \
'learning models using confusion matrices.'

setup(
    name='mcmetrics',
    author='Charlie Sither',
    author_email='toxorhynchitini@gmail.com',
    url='https://github.com/cbsither/monte-carlo-metrics',
    version='0.1.0',
    maintainer='Charlie Sither',
    maintainer_email='toxorhynchitini@gmail.com',
    download_url="https://github.com/cbsither/monte-carlo-metrics.git",
    keywords=['monte carlo', 'prevalence threshold', 'model evaluation', 'machine learning'],
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn'],
    description=written_descsription,
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ]
)