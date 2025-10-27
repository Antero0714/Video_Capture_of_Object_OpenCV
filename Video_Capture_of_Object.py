import cv2
import numpy as np

# Настройки для красного цвета в HSV
LOWER_RED_1 = np.array([0, 100, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([160, 100, 100])
UPPER_RED_2 = np.array([180, 255, 255])

# Минимальная площадь контура
MIN_AREA = 500

def main():
    # 1. Захват видео с камеры
    cap = cv2.VideoCapture(0)
    
    print("'q' для выхода")
    
    while True:
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            break
        
        # Преобразование в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создание маски для красного цвета
        mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Морфологические операции для устранения шума
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. Поиск контуров
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        # Обработка контуров
        if contours:
            # Находим самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Проверяем площадь
            area = cv2.contourArea(largest_contour)
            
            if area > MIN_AREA:
                # 3. Определение области захвата (минимальный прямоугольник)
                rect = cv2.minAreaRect(largest_contour)
                
                # Получаем параметры
                center = rect[0]  # Центр (x, y)
                size = rect[1]    # Размер (width, height)
                angle = rect[2]   # Угол
                
                # Получаем координаты углов прямоугольника
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                # Расчет углов с осями
                angle_horizontal = -angle if angle < 0 else angle
                angle_vertical = 90 - angle_horizontal
                
                # 4. Отрисовка результатов
                
                # Рисуем контур (зеленый)
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                
                # Рисуем минимальный прямоугольник (синий)
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
                
                # Рисуем центр контура (красная точка)
                center_x = int(center[0])
                center_y = int(center[1])
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Рисуем линию области захвата (желтая)
                # Находим середины коротких сторон
                width, height = size
                if width < height:
                    # Короткая сторона горизонтальная
                    mid1 = ((box[0][0] + box[1][0]) // 2, 
                           (box[0][1] + box[1][1]) // 2)
                    mid2 = ((box[2][0] + box[3][0]) // 2, 
                           (box[2][1] + box[3][1]) // 2)
                else:
                    # Короткая сторона вертикальная
                    mid1 = ((box[1][0] + box[2][0]) // 2, 
                           (box[1][1] + box[2][1]) // 2)
                    mid2 = ((box[3][0] + box[0][0]) // 2, 
                           (box[3][1] + box[0][1]) // 2)
                
                cv2.line(frame, mid1, mid2, (0, 255, 255), 2)
                
                # Выводим текстовую информацию
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                text1 = f"Center: ({center_x}, {center_y})"
                cv2.putText(frame, text1, (10, 30), font, 0.6, 
                           (255, 255, 255), 2)
                
                text2 = f"Angle (horiz): {angle_horizontal:.1f} deg"
                cv2.putText(frame, text2, (10, 60), font, 0.6, 
                           (255, 255, 255), 2)
                
                text3 = f"Angle (vert): {angle_vertical:.1f} deg"
                cv2.putText(frame, text3, (10, 90), font, 0.6, 
                           (255, 255, 255), 2)
                
                # Вывод в консоль
                print(f"Центр: ({center_x}, {center_y}) | " +
                      f"Угол гор.: {angle_horizontal:.1f}° | " +
                      f"Угол верт.: {angle_vertical:.1f}°")
            else:
                cv2.putText(frame, "Object too small", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No object detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Отображение результатов
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена.")


if __name__ == "__main__":
    main()
