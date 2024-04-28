from setuptools import setup, find_packages

requirements = [
    "torch",
    "sox",
    "numpy",
    "requests"
]

setup(
    name='fqdd',
    version='v1.0.0',
    installs_package=requirements,  # 表明当前模块依赖哪些包，若环境中没有，则会从pypi中下载安装
    packages=find_packages(),  # 安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    url='https://zw.fqdd.com',
    license='',
    author='fqdd',
    author_email='',
    description=''
)
