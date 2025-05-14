from bs4 import BeautifulSoup
import os
import pandas as pd

# Initialize dictionary with the correct keys
d = {'title': [], 'price': [], 'link': []}

# Loop through HTML files in 'data' directory
for file in os.listdir("data"):
    try:
        with open(f"data/{file}", encoding='utf-8') as f:
            html_doc = f.read()

        soup = BeautifulSoup(html_doc, 'html.parser')

        # Correct tag searching syntax
        t = soup.find("h2")
        title = t.get_text(strip=True) if t else 'N/A'

        # Find price
        p = soup.find("span", attrs={"class": "a-price-whole"})
        price = p.get_text(strip=True) if p else 'N/A'

        # Finding the anchor tag or link within h2
        a_tag = soup.find("a", class_="a-link-normal s-no-outline")
        link = "https://www.amazon.in" + a_tag['href'] if a_tag and a_tag.has_attr('href') else 'N/A'

        # Append to dictionary
        d['title'].append(title)
        d['price'].append(price)
        d['link'].append(link)

    except Exception as e:
        print(f"Error in file {file}: {e}")

# Create DataFrame and save to CSV
df = pd.DataFrame(data=d)
df.to_csv("datanew.csv", index=False)
