import requests
import pandas as pd
import os
import time
import folium
from folium.plugins import HeatMap
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import imageio
import shutil

start = '2014-01-01'
min_mag = 3
lat = 39.1458
lon = 34.1614
max_rad_km = 1000

url = f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start}&minmagnitude={min_mag}&latitude={lat}&longitude={lon}&maxradiuskm={max_rad_km}'
response = requests.get(url)
data = response.json()

df = pd.DataFrame({
    'Place': [feature['properties']['place'] for feature in data['features']],
    'Magnitude': [feature['properties']['mag'] for feature in data['features']],
    'Time': [pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%Y-%m-%d') for feature in data['features']],
    'Latitude': [feature['geometry']['coordinates'][1] for feature in data['features']],
    'Longitude': [feature['geometry']['coordinates'][0] for feature in data['features']]
})

df['Time'] = df['Time'].str.replace(r'-\d{2}$', '-01', regex=True)
"""
Yukarıdaki kodda, verilen parametrelerle USGS API’ını kullanarak deprem verilerini çekiyoruz. 
Başlangıç tarihini (start) '2014-01-01' ve minimum deprem büyüklüğünü (min_mag) 3 olarak belirledik. 
Ayrıca, enlem (lat) 39.1458 ve boylam (lon) 34.1614 olacak şekilde Türkiye’nin koordinatlarına yakın ve 
maksimum 1000 kilometrelik yarıçapa (max_rad_km) sahip bir bölgeyi sorguluyoruz. 
Son olarak, API’dan gelen verileri işleyerek bir DataFrame oluşturuyoruz. 
Bu DataFrame, depremlerin yerini (Place), büyüklüğünü (Magnitude), zamanını (Time) ve enlem-boylam koordinatlarını (Latitude-Longitude) içeriyor.

Aylık bir seri ile çalışacağımız için tarihlerdeki (Time sütunu) günleri 01 ile değiştiriyoruz.
"""

###  Haritaların Yapılması ve HTML-PNG Formatlarında Kaydedilmesi

chrome_options = Options()
chrome_options.add_argument('--start-fullscreen')

unique_dates = df['Time'].unique()
unique_dates.sort()

turkey_latlon = [39, 35]

delay = 5

font = ImageFont.load_default()
font_size = 36

for unique_date in unique_dates:

    filtered_df = df[df['Time'] == unique_date]
    filtered_df = filtered_df[['Latitude', 'Longitude', 'Magnitude']]
    turkey_map = folium.Map(location=turkey_latlon, zoom_start=6, tiles='cartodbdark_matter')
    HeatMap(data=filtered_df, radius=15).add_to(turkey_map)

    html_filename = f'../data/turkey_heatmap_{unique_date}.html'
    turkey_map.save(html_filename)

    browser = webdriver.Chrome(options=chrome_options)
    browser.get(os.path.abspath(html_filename))
    time.sleep(delay)

    screenshot_filename = f'turkey_heatmap_{unique_date}.png'
    browser.save_screenshot(screenshot_filename)

    browser.quit()

    img = Image.open(screenshot_filename)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('arial.ttf', font_size)
    draw.text((10, img.height - 50), pd.to_datetime(unique_date).strftime('%B %Y'), font=font, fill=(255, 255, 255))
    img.save(screenshot_filename)

    os.remove(html_filename)
    print(f'{html_filename} loaded in the browser and screenshot {screenshot_filename} captured.')

    """
    Yukarıdaki kodda, her bir tarih döngüde kullanılacağı için tekil tarihleri unique_dates değişkenine gönderip 
    bu tarihleri eskiden yeniye doğru olacak şekilde sıralıyoruz. Türkiye’nin koordinatlarını tanımladığımız ve 
    haritaların merkezi olacak değerler turkey_latlon değişkeninde bulunuyor. Her işlemde 5 saniyelik bir bekleme süresi olacak. 
    Bunu delay değişkeninde tutuyoruz. Resimlerin üzerinde görünecek yazılara ait varsayılan font ve font büyüklüğü değerleri olan arial.ttf ve 
    36 sırasıyla font ve font_size değişkenlerinde bulunuyor. Her döngüde açılacak tarayıcıların ekranı kapsayacak şekilde olmasını istediğimiz için 
    '--start-fullscreen' olacak şekilde ayarlama da yaptık.

    Döngüde 7 kod grubu bulunmaktadır. 
    İlk grup, tarih filtresi yapıp haritayı oluşturuyor. 
    kinci grup, haritayı içinde bulunduğu dizine HTML formatında kaydediyor. 
    Üçüncü grup, HTML formatında kaydedilen dosyayı tarayıcıda açıyor ve açtıktan sonra belirlenen süre kadar bekletiyor. 
    Dördüncü grup, ekran görüntüsü alıyor ve içinde bulunduğu dizine PNG formatında kaydediyor. 
    Beşinci grup, açılan tarayıcıyı kapatıyor. 
    Altıncı grup, resim üzerinde yazı işlemlerini yapıyor. 
    Yedinci ve son grup, kaydedilen HTML dosyalarını siliyor ve ekrana bilgi veriyor.
    """


##  Animasyonlu Harita Yapımı ve GIF Formatında Kaydedilmesi
image_path = Path()
images = list(image_path.glob('*.png'))
image_list = [imageio.v3.imread(file_name) for file_name in images]
imageio.mimwrite('Turkey_Earthquake.gif', image_list, fps=2)
_ = [file.unlink() for file in images]

shutil.move('Turkey_Earthquake.gif', '../data/Turkey_Earthquake.gif')

"""
Yukarıdaki kodda ilk olarak, Path() fonksiyonunu çağırarak bir dosya yolu nesnesi oluşturuyor ve 
bunu image_path değişkenine atıyoruz. Ardından, bu dosya yolu nesnesi üzerinde .glob() 
yöntemini kullanarak tüm PNG dosyalarını alıyor ve images listesine atıyoruz. imageio modülünü kullanarak her bir PNG dosyasını 
imageio.v3.imread() fonksiyonuyla okuyor ve bu okunan görüntüleri image_list listesine ekliyoruz. imageio.mimwrite() fonksiyonuyla i
mage_list içindeki görüntüleri kullanarak bir GIF dosyası oluşturuyoruz. Oluşturduğumuz GIF dosyasının ismini 'Turkey_Earthquake.gif'
 olarak belirliyor ve saniyede 2 kare (fps=2) hızında olacak şekilde ayarlıyoruz. Daha sonra, artık gereksiz hale
   gelmiş olan PNG dosyalarını tek tek sildiriyoruz. Bu işlem için bir liste dönülüyor ve her bir dosya unlink() yöntemi 
   kullanılarak siliniyor. Son olarak, shutil.move() fonksiyonunu kullanarak oluşturduğumuz GIF dosyasını 'imgs/' dizini altına taşıyoruz.
"""