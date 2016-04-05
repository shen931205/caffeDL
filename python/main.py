#coding=utf-8
'''
Created on 2015年9月23日

@author: shenwc
'''

import re
import urllib


def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html


def getImg(html):
    reg = r'src="(.+?\.jpg)" pic_ext'
    imgre = re.compile(reg)
    imglist = re.findall(imgre,html)
    x = 0
    for img in imglist:
        urllib.urlretrieve(img,'%s.jpg' % x)
        x+=1






if __name__ == '__main__':
    print 'this is my first crawb'
    html = getHtml("http://tieba.baidu.com/p/2460150866")
    print getImg(html)