from setuptools import setup, find_packages

version = '0.5.8'

try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    with open("README.md", "r") as fh:
        long_description = fh.read()
except:
    print('Not finding requirements or readme')
    requirements = ''
    long_description = ''

if __name__ == "__main__":


    setup(name='particle_tracker_one_d',
          version=version,
          description='Particle tracking in Kymographs',
          long_description=long_description,
          long_description_content_type="text/markdown",
          url='',
          author='Johan Tenghamn',
          author_email='tenghamn@chalmers.se',
          license='MIT',
          packages=find_packages(),
          install_requires=requirements,
          keywords='particle tracking nanochannel',
          classifiers=['Development Status :: 4 - Beta', 'Programming Language :: Python :: 3.6', 'Intended Audience :: Science/Research',
                       'Topic :: Scientific/Engineering :: Physics'],
          zip_safe=False)
