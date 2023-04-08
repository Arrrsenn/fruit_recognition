import math
import os
import numpy as np
from skimage import io, feature

# fileTrain = "C:\\Users\\alevo\\PycharmProjects\\lab1\\training"
# fileTest = "C:\\Users\\alevo\\PycharmProjects\\lab1\\testing"
TRAIN_PATH = os.path.dirname(__file__) + "\\training"
TEST_PATH = os.path.dirname(__file__) + "\\testing"
FRUIT1_NAME = os.listdir(TRAIN_PATH)[0]  # Watermelon
FRUIT2_NAME = os.listdir(TEST_PATH)[1]  # Apple Crimson Snow
EPSILON = 1e-8
BLACK_PIXEL = 0
WHITE_PIXEL = 255
CENTER = 49.5


# Преобразуем цветное изображение в черно-белое изображение, применяя пороговое значение яркости
def get_bw_image(image, bw_threshold):
    bw_image = []
    for line in image:
        bw_line = []
        for pixel in line:
            if pixel <= bw_threshold:
                bw_line.append(BLACK_PIXEL)
            else:
                bw_line.append(WHITE_PIXEL)
        bw_image.append(bw_line)
    return np.array(bw_image)


# Нахождениие расстояния между белыми пикселями на изображении и центром изображения, кроме тех пикселей,
# которые попадают в круг в центре изображения заданного радиуса.
def get_distances(image, radius=0):
    distances = []
    for i in range(100):
        for j in range(100):
            if image[i][j] == WHITE_PIXEL:
                distance = math.sqrt(pow(i - CENTER, 2) + pow(j - CENTER, 2))
                if radius and distance >= radius:
                    distances.append(distance)
    return np.array(distances)


# Применяем оператор Canny для выделения границ
# находит расстояния от границ до центра изображения для каждой точки (с помощью функции get_distances),
# находит дисперсию расстояний и возвращает имя типа фрукта, который соответствует этому изображению
def get_predict(path, bw_threshold, variance_threshold, radius=0):
    predict = None
    image = io.imread(path, as_gray=True)
    image = feature.canny(image, sigma=3)
    image = get_bw_image(image, bw_threshold)
    distances = get_distances(image, radius=radius)
    variance = np.var(distances)
    predict = FRUIT1_NAME if (variance > variance_threshold) else FRUIT2_NAME
    return predict


# Вычисление количества верно определенных и неверно
# определенных (ложных) результатов при классификации фруктов на изображении.
def get_tp_fn_or_tn_fp(path, fruit_name, threshold_get_bin, radius,
                       variance_threshold):
    filenames = os.listdir(path + '\\' + fruit_name)
    true_positives_or_true_negatives = 0
    false_negatives_or_false_positives = 0
    for filename in filenames:
        if get_predict(path + '\\' + fruit_name + '\\' + filename,
                       threshold_get_bin,
                       variance_threshold,
                       radius=radius) == fruit_name:
            true_positives_or_true_negatives += 1
        else:
            false_negatives_or_false_positives += 1
    return true_positives_or_true_negatives, false_negatives_or_false_positives


def get_tp_fn_tn_fp(path, threshold_get_bin, radius, variance_threshold):
    true_positives, false_negatives = get_tp_fn_or_tn_fp(path,
                                                         FRUIT1_NAME,
                                                         threshold_get_bin,
                                                         radius,
                                                         variance_threshold)
    true_negatives, false_positives = get_tp_fn_or_tn_fp(path,
                                                         FRUIT2_NAME,
                                                         threshold_get_bin,
                                                         radius,
                                                         variance_threshold)
    return true_positives, false_negatives, true_negatives, false_positives


# Вычисление точности
def get_accuracy(true_positives, true_negatives, false_positives, false_negatives):
    accuracy = (true_positives + true_negatives) / (true_positives +
                                                    true_negatives + false_positives + false_negatives)
    return accuracy


def get_precision(true_positives, false_positives):
    precision = true_positives / (true_positives + false_positives)
    return precision


# Вычисление отрицательного предиктивного значения
def get_negative_predictive_value(true_negatives, false_negatives):
    negative_predictive_value = true_negatives / (true_negatives +
                                                  false_negatives)
    return negative_predictive_value


# Вычисление чувствительности
def get_sensitivity(true_positives, false_negatives):
    sensitivity = true_positives / (true_positives + false_negatives)
    return sensitivity


# Вычисление специфичности
def get_specificity(true_negatives, false_positives):
    specificity = true_negatives
    return specificity


# Нахождение наилучших параметров для классификации изображений фруктов.
# threshold_get_bin - пороговое значение для бинаризации изображения
# radius -  радиус фильтра, который используется для обработки изображения
# variance_threshold - пороговое значение для отбора признаков на основе дисперсии
# current accuracy - текущая точность алгоритма
# best accuracy -  лучшая достигнутая точность алгоритма
def train(path):
    table_caption = '| threshold_get_bin | radius | variance_threshold || current accuracy | best accuracy |'
    line_length = len(table_caption)
    print('_' * line_length)
    print(table_caption)
    print('=' * line_length)

    best_accuracy = None
    threshold_get_bin = 0.1  # Fine tunning
    while best_accuracy != 1 and abs(threshold_get_bin - 0.6) > EPSILON:
        radius = 10  # Fine tunning
        while best_accuracy != 1 and radius != 60:
            variance_threshold = 10  # Fine tunning
            while best_accuracy != 1 and variance_threshold != 100:
                true_positives, false_negatives, true_negatives, false_positives = get_tp_fn_tn_fp(
                    path, threshold_get_bin,
                    radius, variance_threshold
                )
                current_accuracy = get_accuracy(true_positives,
                                                true_negatives,
                                                false_positives,
                                                false_negatives)
                if best_accuracy is None or best_accuracy < current_accuracy:
                    best_accuracy = current_accuracy
                    best_threshold_get_bin = threshold_get_bin
                    best_radius = radius
                    best_variance_threshold = variance_threshold
                if abs(best_accuracy - 1.0) < EPSILON:
                    best_accuracy = 1
                print(
                    '|        {:3.2}        |   {:2}   |         {:2}         ||      {:4.2%}      |     {:4.2%}    |'.format(
                        threshold_get_bin, radius,
                        variance_threshold, current_accuracy,
                        best_accuracy)
                )
                variance_threshold += 10
            print('-' * line_length)
            radius += 10
        print('-' * line_length)
        threshold_get_bin += 0.1
        print('best threshold_get_bin = {}, best radius = {}, best variance_threshold = {}'.format(
            best_threshold_get_bin,
            best_radius,
            best_variance_threshold)
        )
    return best_threshold_get_bin, best_radius, best_variance_threshold


# Вычисление метрик качества на тестовом наборе данных с использованием наилучших найденных параметров
def test(best_threshold_get_bin, best_radius, best_variance_threshold):
    true_positives, false_negatives, true_negatives, false_positives = get_tp_fn_tn_fp(
        TEST_PATH, best_threshold_get_bin,
        best_radius, best_variance_threshold
    )
    accuracy = get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
    precision = get_precision(true_positives, false_positives)
    negative_predictive_value = get_negative_predictive_value(true_negatives, false_negatives)
    sensitivity = get_sensitivity(true_positives, false_negatives)
    specificity = get_specificity(true_negatives, false_positives)
    print(
        'accuracy = {:4.2%}, precision = {:4.2%}, negative predictive value = {:4.2%}, sensitivity = {:4.2%}, '
        'specificity = {:4.2%}'.format(
            accuracy,
            precision,
            negative_predictive_value,
            sensitivity,
            specificity)
    )


# Точка входа в алгоритм
def main():
    print(FRUIT1_NAME)
    best_threshold_get_bin, best_radius, best_variance_threshold = train(TRAIN_PATH)
    print(get_predict(TEST_PATH + '\\85_100.jpg', best_threshold_get_bin, best_variance_threshold,
                      best_radius))


if __name__ == '__main__':
    main()
