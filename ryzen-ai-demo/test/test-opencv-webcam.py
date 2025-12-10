import cv2  
  
def play_webcam():  
    # カメラデバイスを開く（通常、0はデフォルトのカメラデバイスを指します）  
    cap = cv2.VideoCapture(1)
  
    # カメラが正常に開けたか確認  
    if not cap.isOpened():  
        print("Error: カメラを開けませんでした。")  
        return  
  
    # フレームを1つずつ読み込んで表示  
    while True:  
        ret, frame = cap.read()  
  
        # フレームが正常に読み込めなかった場合は終了  
        if not ret:  
            print("Error: フレームを取得できませんでした。")  
            break  
  
        # フレームを表示  
        cv2.imshow('Webcam', frame)  
  
        # 'q'キーが押されたら再生を終了  
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
  
    # リソースを解放  
    cap.release()  
    cv2.destroyAllWindows()  
  
play_webcam()  
