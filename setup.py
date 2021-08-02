from setuptools import setup, find_packages


INSTALL_REQUIREMENTS = ['pytorch torchvision cudatoolkit=10.1 -c pytorch']


INSTALL_REQUIREMENTS = ['numpy>=1.18.1,<1.19', 
	'matplotlib>=3.1.2,<3.2',
	'pandas>=1.0.0,<1.1',
	'scipy>=1.3.2,<1.4',
	'opencv-python>=4.1,<4.2',
	'argparse>=1.1,<1.2',
	'Pillow>=7.0.0,<7.1.0',
	'pydicom>=1.4.1,<1.5',
	'scikit-image>=0.16.2,<0.17',
	'scikit-learn>=0.22,<0.24',
	'tensorboard>=2.1.0,<2.2',
	'torch>=1.4.0+cu101,<1.5.1+cu101',
	'torchvision>=0.5.0+cu101,<0.6.1+cu101',
	'torchsummary>=1.5.1,<1.5.2',
]



setup(
    name                ='f-AnoGAN',
    version             ='1.0',
    description         ='Simple f-AnoGAN implementations for anomaly detection',
    #author_email        ='laura.estacio@1000shapes.com',
    
    install_requires    = INSTALL_REQUIREMENTS,
    packages            = find_packages()
    
)