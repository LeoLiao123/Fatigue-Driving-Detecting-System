# 疲勞偵測系統

這是一個使用 `dlib` 和 `cv2` (OpenCV) 建立的疲勞偵測系統且整合Ubidots和樹梅派來演示功能。它可以偵測使用者的眼睛和嘴巴的動作，以判斷使用者是否感到疲勞。

## Demo & 程式流程

![image](https://github.com/LeoLiao123/Fatigue-Driving-Detecting-System/assets/93932709/55753dad-15da-4922-9f10-405b121aef40)

https://www.youtube.com/watch?v=TBz_t0KP9rc

## 功能

1. **臉部偵測**：使用 `dlib` 和 `cv2` 進行臉部偵測。
2. **特徵點偵測**：偵測臉部的68個特徵點。
3. **疲勞偵測**：通過眼睛和嘴巴的動作來判斷疲勞。
4. **警告系統**：當偵測到使用者疲勞時，會發出警告。
5. **Ubidots 整合**：將疲勞數據上傳到 Ubidots 平台並寄送警告郵件。

## 安裝 

1. 安裝所需的 Python 庫：

```bash
pip install dlib opencv-python ubidots speech_recognition RPi.GPIO gTTS
```

```bash
pip install pyttsx3
```

2. 訓練辨識模型：

本程式驗證使用者階段所訓練的模型是基於 https://steam.oxxostudio.tw/category/python/ai/ai-face-recognizer.html 這個教學所訓練出來的，透過各個使用者提供各100張的自拍影像以及適當的Augmentation( 隨機翻轉、灰階以及ColorJitter )來小幅擴充數據集，並將其與程式放在同個目錄。

這個教學使用的模型是一個特徵分類器精準度相較於目前的深度學習網路來說較為差勁，可以自行尋找適合的網路跟預訓練模型。

3. 註冊Ubidots網站

-取得API金鑰

-根據設置的使用者數量創建容器

-建立寄送郵件事件並連接容器做數據分析以此界定事件是否發生


## 使用方法

```bash
python your_code_name.py
```

1. 程式將開始偵測使用者的臉部，並分析是否有疲勞的跡象。
2. 若偵測到疲勞，會發出警告並透過 LED 提醒使用者。

## 注意事項

確保 Raspberry Pi 已正確連接 LED。

請在光線充足的地方使用此系統，以提高偵測的準確性。


## 參考資料


https://steam.oxxostudio.tw/category/python/ai/ai-face-recognizer.html

http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

[https://blog.csdn.net/cungudafa/article/details/103496881](https://www.cnblogs.com/lushuang55/p/17396900.html)

