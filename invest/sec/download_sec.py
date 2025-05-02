import pdb
from sec_downloader import Downloader
from bs4 import BeautifulSoup
import sec_parser as sp
import re

def replace_multiple_newlines(text):
  """Replaces multiple line breaks with a single line break."""
  res = re.sub(r'\n\s+', '\n', text)
  res = re.sub(r'\n+', '\n', res)
  return res 

# Utility function to make the example code a bit more compact
def print_first_n_lines(text: str, *, n: int):
    print("\n".join(text.split("\n")[:n]), "...", sep="\n")

# Initialize the downloader with your company name and email
dl = Downloader("MyCompanyName", "email@example.com")

# Download the latest 10-Q filing for Apple
metadatas = dl.get_filing_metadatas("5/MSFT/10-K")
html_list = []

for metadata in metadatas:
    html = dl.download_filing(url=metadata.primary_doc_url)
    html_list.append(html)
    elements: list = sp.Edgar10QParser().parse(html)

    demo_output: str = sp.render(elements)
    print_first_n_lines(demo_output, n=7)

    all_tx = ''
    for element in elements: 
        tx = replace_multiple_newlines(element.text)
        if len(tx) > 500: 
            all_tx += (tx.strip() + '\n')
    
    f = open('test.txt', 'w')
    f.write(all_tx)
    f.close()
    pdb.set_trace()
    
    """
    soup = BeautifulSoup(html, features="html.parser")
    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    """
    
    


