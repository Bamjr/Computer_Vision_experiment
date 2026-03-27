import cv2
import mediapipe as mp
import webbrowser
import urllib.parse
import time

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands = 1,    min_detection_confidence=0.7,
    min_tracking_confidence=0.7 )

tip_ids = [4, 8, 12, 16, 20]

last_search_time = 0
cooldown_seconds = 1.5

last_count = None
stable_count = 0
stable_threshold = 6

search_map = {
    1: "person pointing one finger meme",
    2: "person showing two fingers meme",
    3: "person showing three fingers meme",
    4: "person showing four fingers meme",
    5: "open hand meme"
}

def count_fingers(lm_list):
    """
    lm_list = [(x,y), ...] จำนวน 21 จุด
    คืนค่า:
      finger_count, finger_states
    finger_states = [thumb, index, middle, ring, pinky]
    """
    if len(lm_list) != 21:
        return 0, [0, 0, 0, 0, 0]

    fingers = []


    if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)


    for i in range(1, 5):
        if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers), fingers



while True:
    _, frm = cap.read()

    frm = cv2.flip(frm , 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    res = hand_obj.process(rgb)

    finger_count = 0
    gesture_text = "No hand"

    

    if res.multi_hand_landmarks:
        hand_landmarks = res.multi_hand_landmarks[0]
        drawing.draw_landmarks(frm, hand_landmarks, hands.HAND_CONNECTIONS)

        h, w, _ = frm.shape
        lm_list = []

        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))

        finger_count, finger_states = count_fingers(lm_list)
       
        gesture_text = f"Fingers: {finger_count}"

        if finger_count == last_count:
            stable_count += 1
        else:
            stable_count = 0
            last_count = finger_count

        current_time = time.time()

       
        if (
            stable_count >= stable_threshold
            and finger_count in search_map
            and current_time - last_search_time > cooldown_seconds
        ):
            query = search_map[finger_count]
            encoded_query = urllib.parse.quote(query)

            # ค้นหารูปใน Google Images
            url = f"https://www.google.com/search?tbm=isch&q={encoded_query}"
            webbrowser.open(url)

            print(f"Searching: {query}")
            last_search_time = current_time
            stable_count = 0  # reset กันเปิดซ้ำ

    else:
        last_count = None
        stable_count = 0

    # แสดงผลบนจอ
    cv2.putText(frm, gesture_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frm, f"Stable: {stable_count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frm, "ESC = Exit", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Hand Meme Search", frm)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

