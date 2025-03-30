import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO("/home/wonsuk1025/kai_ws/src/cam_yolo/runs/detect/train4/weights/best.pt")

# 클래스별 색상 정의 (예시: blue, red, yellow)
class_colors = {
    'blue cone': (255, 0, 0),    # 파란색
    'red cone': (0, 0, 255),     # 빨간색
    'yellow cone': (0, 255, 255) # 노란색
}

# 웹캠 열기
cap = cv2.VideoCapture('/dev/video4')  # 카메라 경로 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다!")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 카메라 열기 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 웹캠에서 프레임을 읽기
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 모델로 객체 감지
    results = model(frame)

    # 결과에서 각 객체의 정보 추출 (box, confidence, class_id 등)
    for result in results[0].boxes:
        box = result.xywh[0].cpu().numpy()  # 바운딩 박스 좌표
        class_id = int(result.cls[0].cpu().numpy())  # 클래스 ID
        confidence = result.conf[0].cpu().numpy()  # 신뢰도

        # 클래스 이름
        class_name = results[0].names[class_id]

        # 색상 설정 (클래스별 색상)
        color = class_colors.get(class_name, (255, 255, 255))  # 기본값은 흰색

        # 바운딩 박스 그리기
        x1, y1, w, h = box
        x1, y1, w, h = int(x1 - w / 2), int(y1 - h / 2), int(w), int(h)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        # 클래스 이름과 신뢰도 표시
        label = f'{class_name} {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 출력
    cv2.imshow("YOLOv8 Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제
cap.release()
cv2.destroyAllWindows()
