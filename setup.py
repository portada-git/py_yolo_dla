from setuptools import setup

setup(name='py_yolo_dla',
      version='0.0.5',
      description='Process to get columns from documment images using a trained YOLO model in PortADa project',
      author='PortADa team',
      author_email='jcbportada@gmail.com',
      license='MIT',
      url="https://github.com/portada-git/py_yolo_dla",
      packages=['py_yolo_dla'],
      py_modules=['py_yolo_dla_page'],
      install_requires=[
          'urllib3',
          'opencv-python',
          'numpy < 2',
          'ultralytics',
      ],
      python_requires='>=3.9',
      zip_safe=False)
