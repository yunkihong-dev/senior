import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import onnxruntime as ort
import numpy as np
import av

# RTC 설정 (여기서는 기본값을 사용합니다)
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class_names = {0: 'normal', 1: 'falling', 2: 'wandering'}

# ONNX 모델 로드
ort_session = ort.InferenceSession("runs/train/senior4/weights/best.onnx")

class VideoTransformer(VideoProcessorBase):
    async def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 모델 입력을 준비
        resized = cv2.resize(img, (640, 640))
        input_data = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)

        # ONNX 모델 실행
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_data})

        # NMS 전 준비: 바운딩 박스, 신뢰도 점수, 클래스 ID 추출
        boxes = []
        confidences = []
        class_ids = []
        for detection in outputs[0][0]:
            x1, y1, x2, y2, conf, class_id = detection[:6]
            if conf >= 0.3:
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # NMSBoxes 함수를 위한 형식
                confidences.append(float(conf))
                class_ids.append(int(class_id))

        # Non-Maximum Suppression 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)

        # NMS를 통과한 탐지만 그리기
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x1, y1, x2, y2 = map(int, [x, y, x + w, y + h])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[class_ids[i]]} {confidences[i]:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 처리된 이미지를 다시 av.VideoFrame 객체로 변환
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        return new_frame


def main():
    st.title("실시간 웹캠 ONNX 객체 탐지")

    # 웹캠 스트리밍
    webrtc_streamer(key="example", 
                    video_processor_factory=VideoTransformer,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False})

if __name__ == "__main__":
    main()
