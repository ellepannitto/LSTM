from setuptools import setup

setup(
    name='lstm',
    description='build lstm',
    author=['Ludovica Pannitto'],
    author_email=['ellepannitto@gmail.com'],
    url='https://github.com/ellepannitto/LSTM',
    version='0.1.0',
    license='MIT',
    packages=['lstm', 'lstm.logging_utils', 'lstm.utils', 'lstm.core'],
    package_data={'lstm': ['logging_utils/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'lstm = lstm.main:main'
        ],
    },
    install_requires=[],
)