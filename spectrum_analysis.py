"""
Скрипт для сравнения УФ-ВИС спектров.
Автоматически находит все файлы известных и неизвестных веществ,
сравнивает их методом кросс-корреляции с нормализацией.
"""

import pandas as pd
import numpy as np
import scipy.signal as signal
import glob
import os
import matplotlib.pyplot as plt

def load_spectrum(filepath):
    """
    Загружает спектр из CSV файла.
    Формат: разделитель столбцов - ';', десятичный разделитель - ','.
    Возвращает два массива: длины волн и оптические плотности.
    """
    try:
        # Чтение CSV с указанием разделителей
        df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8')
        
        # Проверка наличия нужных столбцов
        if 'wavelength' not in df.columns or 'absorbance' not in df.columns:
            raise ValueError(f"Файл {filepath} должен содержать столбцы 'wavelength' и 'absorbance'")
        
        wavelengths = df['wavelength'].values
        absorbance = df['absorbance'].values
        
        # Проверка диапазона и шага
        if len(wavelengths) != 511:  # 190-700 нм с шагом 1 нм = 511 точек
            print(f"  Внимание: {filepath} содержит {len(wavelengths)} точек, ожидается 511")
        
        return wavelengths, absorbance
        
    except Exception as e:
        print(f"Ошибка при чтении {filepath}: {e}")
        return None, None

def normalize_spectrum(absorbance, method='max'):
    """
    Нормализует спектр для устранения влияния концентрации.
    
    Параметры:
    - absorbance: массив оптических плотностей
    - method: 'max' - нормализация по максимуму, 
              'area' - нормализация по площади под кривой
    
    Возвращает нормализованный спектр.
    """
    if method == 'max':
        # Нормализация по максимуму
        max_val = np.max(absorbance)
        if max_val > 0:
            return absorbance / max_val
        else:
            return absorbance
    elif method == 'area':
        # Нормализация по площади (интегралу)
        area = np.trapz(absorbance)
        if area > 0:
            return absorbance / area
        else:
            return absorbance
    else:
        raise ValueError("Метод нормализации должен быть 'max' или 'area'")

def compare_spectra(unknown_abs, known_abs):
    """
    Сравнивает два спектра с помощью кросс-корреляции.
    
    Возвращает коэффициент сходства (0-1), где 1 - идеальное совпадение.
    """
    # Нормализуем оба спектра (по максимуму, как обсуждали)
    unknown_norm = normalize_spectrum(unknown_abs, method='max')
    known_norm = normalize_spectrum(known_abs, method='max')
    
    # Вычисляем кросс-корреляцию
    correlation = signal.correlate(unknown_norm, known_norm, mode='same')
    
    # Нормируем максимальное значение корреляции
    max_correlation = np.max(correlation)
    norm_factor = np.sqrt(np.sum(unknown_norm**2) * np.sum(known_norm**2))
    
    if norm_factor > 0:
        similarity = max_correlation / norm_factor
    else:
        similarity = 0
    
    return round(similarity, 4)

def analyze_unknown(unknown_name, unknown_path, known_files):
    """
    Анализирует одно неизвестное вещество против всех известных.
    
    Возвращает отсортированный список результатов.
    """
    print(f"\n{'='*60}")
    print(f"Анализ: {unknown_name}")
    print(f"{'='*60}")
    
    # Загружаем спектр неизвестного вещества
    unknown_wl, unknown_abs = load_spectrum(unknown_path)
    if unknown_wl is None:
        print(f"Не удалось загрузить {unknown_name}")
        return []
    
    results = []
    
    # Сравниваем со всеми известными веществами
    for known_name, known_path in known_files:
        # Загружаем спектр известного вещества
        known_wl, known_abs = load_spectrum(known_path)
        if known_wl is None:
            continue
        
        # Проверяем совпадение диапазонов длин волн
        if len(unknown_wl) != len(known_wl):
            print(f"  Внимание: разное количество точек в {known_name}")
            # Можно добавить интерполяцию, но пока просто предупреждаем
        
        # Сравниваем спектры
        similarity = compare_spectra(unknown_abs, known_abs)
        
        # Добавляем результат
        results.append({
            'known_substance': known_name,
            'similarity': similarity,
            'file': known_path
        })
        
        print(f"  {known_name}: {similarity:.4f}")
    
    # Сортируем по убыванию сходства
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results

def plot_comparison(unknown_name, unknown_wl, unknown_abs, 
                    best_match_name, best_match_wl, best_match_abs):
    """
    Создает визуализацию сравнения спектров.
    """
    plt.figure(figsize=(10, 6))
    
    # Нормализуем для сравнения формы
    unknown_norm = normalize_spectrum(unknown_abs, 'max')
    best_match_norm = normalize_spectrum(best_match_abs, 'max')
    
    plt.plot(unknown_wl, unknown_norm, 'b-', linewidth=2, 
             label=f'{unknown_name} (неизвестное)')
    plt.plot(best_match_wl, best_match_norm, 'r--', linewidth=2, 
             label=f'{best_match_name} (наиболее похожее)')
    
    plt.xlabel('Длина волны, нм', fontsize=12)
    plt.ylabel('Нормализованная оптическая плотность', fontsize=12)
    plt.title(f'Сравнение: {unknown_name} vs {best_match_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем график
    filename = f"comparison_{unknown_name}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  График сохранен как: {filename}")

def main():
    """
    Главная функция программы.
    """
    print("\n" + "="*60)
    print("Анализ УФ-ВИС спектров: сравнение неизвестных и известных веществ")
    print("="*60)
    
    # Автоматический поиск файлов
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Находим файлы неизвестных веществ
    unknown_files = []
    for file in glob.glob(os.path.join(current_dir, "unknown_*.csv")):
        name = os.path.basename(file).replace('.csv', '').replace('unknown_', '')
        unknown_files.append((name, file))
    
    # Находим файлы известных веществ
    known_files = []
    for file in glob.glob(os.path.join(current_dir, "known_*.csv")):
        name = os.path.basename(file).replace('.csv', '').replace('known_', '')
        known_files.append((name, file))
    
    if not unknown_files:
        print("ОШИБКА: Не найдены файлы unknown_*.csv в текущей папке!")
        print("Убедитесь, что файлы названы правильно:")
        print("  unknown_red.csv, unknown_violet.csv")
        return
    
    if not known_files:
        print("ОШИБКА: Не найдены файлы known_*.csv в текущей папке!")
        print("Убедитесь, что файлы названы правильно:")
        print("  known_Gencianviolet.csv, known_Etilviolet.csv и т.д.")
        return
    
    print(f"\nНайдено неизвестных веществ: {len(unknown_files)}")
    print(f"Найдено известных веществ: {len(known_files)}")
    
    all_results = {}
    
    # Анализируем каждое неизвестное вещество
    for unknown_name, unknown_path in unknown_files:
        results = analyze_unknown(unknown_name, unknown_path, known_files)
        all_results[unknown_name] = results
        
        # Выводим ранжированные результаты
        if results:
            print(f"\nРанжированные результаты для {unknown_name}:")
            print("-" * 40)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['known_substance']}: {result['similarity']:.4f}")
            
            # Визуализация с наиболее похожим веществом
            try:
                best = results[0]
                unknown_wl, unknown_abs = load_spectrum(unknown_path)
                best_wl, best_abs = load_spectrum(best['file'])
                
                plot_comparison(unknown_name, unknown_wl, unknown_abs,
                              best['known_substance'], best_wl, best_abs)
            except Exception as e:
                print(f"  Не удалось создать график: {e}")
    
    # Сохраняем результаты в CSV
    save_results_to_csv(all_results)
    
    print("\n" + "="*60)
    print("Анализ завершен!")
    print("="*60)

def save_results_to_csv(all_results):
    """
    Сохраняет все результаты в CSV файл.
    """
    try:
        output_data = []
        for unknown_name, results in all_results.items():
            for result in results:
                output_data.append({
                    'unknown_substance': unknown_name,
                    'known_substance': result['known_substance'],
                    'similarity': result['similarity']
                })
        
        if output_data:
            df_output = pd.DataFrame(output_data)
            df_output.to_csv('spectrum_comparison_results.csv', 
                           sep=';', decimal=',', index=False, encoding='utf-8')
            print(f"\nРезультаты сохранены в: spectrum_comparison_results.csv")
    except Exception as e:
        print(f"Не удалось сохранить результаты: {e}")

# Запуск программы
if __name__ == "__main__":
    main()