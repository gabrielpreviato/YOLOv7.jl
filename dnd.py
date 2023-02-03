# /html/body/div[1]/div[2]/div[1]/div[1]/div/span[1]/div[1]/div[2]/div/div/div/div/div[3]/div[3]/div[2]/div[1]/a/span/span[1]

# /html/body/div[1]/div[2]/div[1]/div[1]/div/span[1]/div[1]/div[3]

# /html/body/div[1]/div[2]/div[1]/div[1]/div/span[1]/div[1]

# from lxml import html
# import requests
# import time

# page = requests.get('https://www.amazon.com.br/s?k=Wizards+of+the+Coast&ref=bl_dp_s_web_16194299011')

# tree = html.fromstring(page.content)

# trs = tree.xpath('//*[@id="search"]/div[1]/div[1]/div/span[1]/div[1]')[0]
# print("trs:", trs)

# for i, tr in enumerate(trs.getchildren()):
#     time.sleep(0.1)
#     if i <= 1:
#         continue
#     # print(tr)
#     # b = '//*[@id="search"]/div[1]/div[1]/div/span[1]/div[1]/div[%d]/div/div/div/div/div[3]/div[3]/div[2]/div[1]/a/span/' % i
#     c = '//*[@id="search"]/div[1]/div[1]/div/span[1]/div[1]/div[%d]/div/div/div/div/div[3]/div[3]/div[2]/div[1]/a/span/span[1]' % i
#         # '//*[@id="search"]/div[1]/div[1]/div/span[1]/div[1]/div[2]/div/div/div/div/div[3]/div[3]/div[2]/div[1]/a/span/span[1]'
#     d = '//*[@id="search"]/div[1]/div[1]/div/span[1]/div[1]/div[%d]/div/div/div/div/div[2]/div[3]/div[2]/div[1]/a/span/span[1]' % i
#     print(c)
#     a = tree.xpath(c)
#     if len(a) > 0:
#         print(a[0].text)
#     a = tree.xpath(d)
#     if len(a) > 0:
#         print(a[0].text)

import requests
import json
from bs4 import BeautifulSoup

url = 'https://www.amazon.com.br/s?k=Wizards+of+the+Coast&ref=bl_dp_s_web_16194299011'
res = requests.get(url)
html = BeautifulSoup(res.text, 'lxml')

spans = html.find_all('span', {'class': 'a-offscreen'})
for span in spans:
    print(span.text)