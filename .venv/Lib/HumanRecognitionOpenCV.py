import cv2
import torch
from ultralytics import YOLO

# Modeli CPU üzerinde çalışacak şekilde yükle
device = 'cpu'

# Eğitimli YOLO modelini yükle (best.pt modelini kendi modelinle değiştir)
model = YOLO("best.pt").to(device)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Kamera ayarlarını belirle
cap.set(3, 1280)  # Görüntü genişliği
cap.set(4, 720)   # Görüntü yüksekliği
cap.set(5, 30)    # Kare hızı

# Sabit değerler (mesafe hesaplaması için)
avg_human_height = 180.0  # Ortalama insan boyu (cm)
focal_length = 1280  # Kameranın odak uzunluğu
avg_ratio = 0.25  # Omuz genişliği ile boy oranı

# Mesafe hesaplama fonksiyonu
def calculate_distance(pixel_height, pixel_width):
    estimated_height = pixel_width / avg_ratio  # Piksel boyutlarından tahmini boy hesaplama
    return (avg_human_height * focal_length) / estimated_height  # Mesafe hesaplama formülü

# Ekranı gösterme penceresini ayarla
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Uyarı mesafesi
warning_threshold = 300  # Uyarı mesafesi (cm)

while cap.isOpened():
    ret, frame = cap.read()  # Kameradan bir kare oku
    if not ret:
        break

    # Giriş görüntüsünü modele uygun boyuta getir (YOLO genellikle 640x640 kullanır)
    resized_frame = cv2.resize(frame, (640, 640))

    # Giriş görüntüsünü tensöre çevir ve normalleştir
    frame_tensor = torch.from_numpy(resized_frame).to(device).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW formatına dönüştürme

    # Modeli kullanarak insanları algıla
    with torch.no_grad():  # Gradient hesaplamalarını devre dışı bırak
        results = model(frame_tensor)

    warning_displayed = False  # Her karede uyarı mesajını sıfırla
    if len(results) > 0:
        result = results[0]

        for i, (x1, y1, x2, y2) in enumerate(result.boxes.xyxy):
            label = result.names[int(result.boxes.cls[i])]
            confidence = result.boxes.conf[i]
            pixel_height = y2 - y1  # Nesnenin piksel yüksekliğini hesapla
            pixel_width = x2 - x1  # Nesnenin piksel genişliğini hesapla

            distance = calculate_distance(pixel_height, pixel_width)  # Mesafeyi hesapla

            # Algılanan nesnenin etrafına dikdörtgen çiz ve mesafeyi göster
            cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{label} {confidence:.2f}, Distance: {distance:.2f}cm", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Mesafe belirlenen uyarı mesafesinden küçükse uyarı mesajı göster
            if distance < warning_threshold and not warning_displayed:
                cv2.putText(resized_frame, "! TOO CLOSE!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                warning_displayed = True

    # İşlenen kareyi göster
    cv2.imshow('Output', resized_frame)

    # 'q' tuşuna basıldığında döngüyü kır ve çık
    if cv2.waitKey(33) == ord('q'):
        break

# Kamera ve pencereleri serbest bırak
cap.release()
cv2.destroyAllWindows()
