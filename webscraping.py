from bs4 import BeautifulSoup
import requests


def extract(job):
    header={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
    url=f'https://www.postjobfree.com/jobs?q={job}&l=India&radius=25'
    r=requests.get(url,header)
    #return r.status_code
    soup=BeautifulSoup(r.content,'lxml')
    return soup

def transform(soup):
    divs =soup.find_all('div',class_='snippetPadding')
    for item in divs:
        title=item.find('a').text
        company=item.find('span',class_='colorCompany').text
        job={
            'title':title,
            'company':company
        }
        print(job)
        joblist.append(job)
    return

joblist=[]

