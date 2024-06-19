import cv2
import numpy as np

# Kullanacağımız videonun adresini yazıyoruz. Webcam için 0 yazıyoruz
kamera= cv2.VideoCapture('../data/video.mp4')

# Yazı fontunu tanımlıyoruz.
font = cv2.FONT_HERSHEY_DUPLEX

# Her rengin yer aldığı bir HSV değer aralığı var.
# Kırmızı renk için bu değerleri tanımlıyoruz.
low_red = np.array([170, 70, 50])
high_red = np.array([180, 255, 255])

# Toplam kaç şişenin geçtiğini tutacağımız değişken
count=0

# Şişenin geçip geçmediğini kontrol edeceğimiz değişken
control=False

while True:
    
  
        
    # Kameradan gelen görüntüyü alıyoruz.
    # Kameradan gelen görüntü biterse ret False olur
    ret,frame=kamera.read()
    if not ret:
        break
   
    # Gelen görüntüyü HSV renk uzayına çeviriyoruz.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Görüntüye bir filtreleme uyguluyoruz. Kırmızı renk değerlerini kullandığımız için görüntüde kırmızı olan yerler beyaz
    # geri kalan renklere sahip pikseller siyah oldu.
    # Bizim sayacağımız şişeler kırımızı renkte olduğu için böyle bir şey yaptık. Buradaki beyaz alanlar kırmızı şişeler olacak
    mask = cv2.inRange(hsv_frame, low_red, high_red)
    
    # Maskeye erezyon, aşındırma işlemi uyguluyoruz. Böylece beyaz kısımlar daha keskinleşmiş oluyor.
    mask = cv2.erode(mask,None,iterations=2)
    
    # Maskeye blur işlemi uyguluyoruz
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Maskedeki beyaz olan yerleri koordinatlarını buluyrouz.
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    #Burada yukarıdan aşağıya bir çizgi çekiyoruz. Bu çizgi görsel amaçlı kullanılıyor.
    # Şişelerin nereden geeçtikten sonra geçti kabul edildiğini biz beliyoryoruz.
    # O yüzden bu koordinatlar aşağıdaki koordinatlar ile uyumlu olmalı.
    # Burada benim belirdiğim çizgi yatayda 700. piksel. 
    cv2.line(frame,(700,0),(700,720),(255,0,0), 10)
    
    # Burada her bir alan için konumlara bakıyoruz.
    for cnt in cnts:
        
        # Burada alanın konumunu dikdörtgen formatta aldık.
        # x ve y sol üst köşe w dikdörtgenin genişliği h ise yüksekliği
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Çok küçük kırmızı alanları göz ardı etmek için belirli büyüklükteki alanları dikkate alıyoruz
        if w>100 and h>200 :

            # alanın yani şişenin orta noktaları
            cx=int(x+w/2)
            cy=int(y+h/2)
            
            # Şişenin geçip geçmediğini kontrol etmek için görsel üzerinden bir bölge seçiyoruz
            # Ben 700. pikselde bakacağım için bölge olarak bu noktadan 50 geri ve 50 ileri şeklinde bir bölge seçtim
            # Şişenin orta noktası bu bölgeye gelmişse geçilip geçilmediği için kontrole alınacak
            if cx>650 and cx<750:
                
                # Eğer şişe bölgenin içinde ise geçiş noktasının solunda mı diye kontrol ediyoruz.
                # Eğer böyle bir kontrol yapmazsak şişenin orta noktası birden fazla bu bölgeye girdiği için 
                # bir şişe birden fazla sayılacaktır. 
                # Şişe eğer geçiş noktasının solunda ise control değişkenini True yapıyrouz.
                if cx<700:
                    control=True
                
                # Şişe geçiş noktasının sağında mı diye bakıyoruz.Ayrıcca control değişkenine da bakıyoruz.
                # Eğer sağında ise ve control True ise şişeyi geçti kabul edip count değişkenini 1 arttıryoruz.
                # Ayrıca control değişkenini false yapıyoruz.
                # Böylece bir sonraki kısımda hala sağında olsa bile bu if bloğunun içine girlmeyeceği için
                # bir şişe sadece bir kere sayılacak
                if cx>700 and control:
                    control=False
                    count+=1
                    
                    # Geçiş esnasında çizginin rengini sarı yapıyoruz.
                    cv2.line(frame,(700,0),(700,720),(0,255,255), 10)
                
               
                # Şişe bölgede iken orta noktasını beyaz bir daire gösteriyoruz.
                cv2.circle(frame,(cx,cy),20,(255,255,255),-1)
    
    # Toplam kaç şişe geçtiğini ekranda gösteriyoruz
    strcount= "Toplam: "+str(count)
    cv2.putText(frame,strcount ,(0, 60), font, 2, (102,0,153), 2)  
    
    
    cv2.imshow("kamera",frame)
  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release() 
cv2.destroyAllWindows()