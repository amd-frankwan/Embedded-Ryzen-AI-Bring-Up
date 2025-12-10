import cv2  
  
def play_video(video_path):  
    # 動画ファイルを読み込む  
    cap = cv2.VideoCapture(video_path)  
  
    # 動画が正常に読み込めたか確認  
    if not cap.isOpened():  
        print("Error: 動画ファイルを開けませんでした。")  
        return  
  
    # フレームを1つずつ読み込んで表示  
    while True:  
        ret, frame = cap.read()  
  
        # フレームが正常に読み込めなかった場合は終了  
        if not ret:  
            break  
  
        # フレームを表示  
        cv2.imshow('Video', frame)  
  
        # 'q'キーが押されたら再生を終了  
        if cv2.waitKey(25) & 0xFF == ord('q'):  
            break  
  
    # リソースを解放  
    cap.release()  
    cv2.destroyAllWindows()  
  
# 動画ファイルのパスを指定  
video_path = r"c:\Users\juna\OneDrive - Advanced Micro Devices Inc\Alveo\Demo\vvas-v70-demo\v1.0\config\pose-estimation-2\videos\30537_japanese.mp4.ts"
play_video(video_path)
