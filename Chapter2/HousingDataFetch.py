"""
    date:       2019/12/27 7:06 下午
    written by: neonleexiang
"""


"""
    we have been extract the data from the source repository
    and also we extract the data into csv file
    
    This python file is to change the jupyter notebook file into py
    and first thing to do is to fetch the data from github repository
"""


# import list
import os
import tarfile
from six.moves import urllib


# parameters path setting
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"   # we fetch the data from the repository
HOUSING_PATH = "datasets/housing"   # also use to set the path of storing
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# function for data fetching
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):     # create path for storing
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)   # search the data by using url and request it
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)   # extract the data into csv file
    housing_tgz.close()


"""
函数：urllib.urlretrieve(url[, filename[, reporthook[, data]]]) 

参数说明： 

url：外部或者本地url ,url中不要含有中文，好像会出错。

filename：指定了保存到本地的路径（如果未指定该参数，urllib会生成一个临时文件来保存数据）； 

reporthook：是一个回调函数，当连接上服务器、以及相应的数据块传输完毕的时候会触发该回调。我们可以利用这个回调函数来显示当前的下载进度。 

data：指post到服务器的数据。该方法返回一个包含两个元素的元组(filename, headers)，filename表示保存到本地的路径，header表示服务器的响应头。
 
"""

"""
the above function is to collect data and change the data into csv file

once you use the function, it will automatically create a dir datasets/housing
to store the tgz file name:housing.tgz, and up zipper it into csv file

"""

"""
we found that it occur an error call:

URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:749)>


so we search the solution and get the solution code below:

from the stackoverflow it said: a dirty but fast hack

"""


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == '__main__':
    fetch_housing_data()

