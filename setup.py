from setuptools import find_packages, setup
from glob import glob

package_name = 'dialog_example'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),  # ✅ This line installs your launch files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alon',
    maintainer_email='alonborn@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
    'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'dialog_node = dialog_example.dialog_node:main',  # Correct entry point for your script
            'ov5640_publisher = dialog_example.ov5640_publisher:main',
        ],
}
)
