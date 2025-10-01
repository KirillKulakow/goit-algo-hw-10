import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import scipy.special as sps
from typing import Callable, Tuple
import time


def monte_carlo_integration(fn: Callable, low: float, high: float, n_points: int = 100000) -> Tuple[
    float, np.ndarray, np.ndarray]:
    """
    Обчислює визначений інтеграл функції методом Монте-Карло.

    Args:
        fn: Функція для інтегрування
        low: Нижня межа інтегрування
        high: Верхня межа інтегрування
        n_points: Кількість випадкових точок для генерації

    Returns:
        Tuple: (результат інтеграла, x-координати точок, y-координати точок)
    """
    # Генеруємо випадкові точки
    x_random = np.random.uniform(low, high, n_points)
    y_random = fn(x_random)

    # Обчислюємо середнє значення функції
    avg_value = np.mean(y_random)

    # Обчислюємо інтеграл: площа = середнє значення * ширина інтервалу
    integral = avg_value * (high - low)

    return integral, x_random, y_random


def monte_carlo_with_visualization(fn: Callable, low: float, high: float,
                                   n_points: int = 10000,
                                   show_points: bool = True,
                                   func_name: str = "f(x)") -> float:
    """
    Метод Монте-Карло з візуалізацією випадкових точок.

    Args:
        fn: Функція для інтегрування
        low: Нижня межа
        high: Верхня межа
        n_points: Кількість точок
        show_points: Чи показувати випадкові точки на графіку
        func_name: Назва функції для відображення

    Returns:
        Значення інтеграла
    """
    # Знаходимо максимальне та мінімальне значення функції на інтервалі
    x_test = np.linspace(low, high, 1000)
    y_test = fn(x_test)
    y_max = max(y_test) * 1.1  # Додаємо 10% запасу
    y_min = min(0, min(y_test) * 1.1)  # Враховуємо від'ємні значення

    # Генеруємо випадкові точки в прямокутнику
    x_random = np.random.uniform(low, high, n_points)
    y_random = np.random.uniform(y_min, y_max, n_points)

    # Обчислюємо значення функції в точках x
    y_func = fn(x_random)

    # Визначаємо точки під/над кривою
    # Для додатних значень функції - точки між 0 та f(x)
    # Для від'ємних значень функції - точки між f(x) та 0
    under_curve = np.where(y_func >= 0,
                           (y_random >= 0) & (y_random <= y_func),
                           (y_random <= 0) & (y_random >= y_func))

    # Обчислюємо інтеграл
    area_rectangle = (high - low) * (y_max - y_min)
    positive_points = np.sum((y_random >= 0) & under_curve)
    negative_points = np.sum((y_random < 0) & under_curve)
    integral = area_rectangle * (positive_points - negative_points) / n_points

    if show_points:
        # Візуалізація
        fig, ax = plt.subplots(figsize=(12, 7))

        # Малюємо функцію
        x_plot = np.linspace(low - 0.5, high + 0.5, 1000)
        y_plot = fn(x_plot)
        ax.plot(x_plot, y_plot, 'b-', linewidth=2.5, label=func_name, zorder=5)

        # Показуємо випадкові точки (тільки підвибірку для кращої візуалізації)
        sample_size = min(2000, n_points)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)

        # Точки під/в області кривої - зелені
        under_indices = sample_indices[under_curve[sample_indices]]
        ax.scatter(x_random[under_indices], y_random[under_indices],
                   c='green', s=0.5, alpha=0.4, label='В області інтегрування')

        # Точки поза областю - червоні
        over_indices = sample_indices[~under_curve[sample_indices]]
        ax.scatter(x_random[over_indices], y_random[over_indices],
                   c='red', s=0.5, alpha=0.4, label='Поза областю')

        # Заповнення області під кривою
        x_fill = np.linspace(low, high, 500)
        y_fill = fn(x_fill)
        ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.3, label='Область інтегрування')

        # Налаштування графіка
        ax.set_xlim((low - 0.5, high + 0.5))
        ax.set_ylim((y_min, y_max))
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title(f'Метод Монте-Карло: {n_points:,} точок\nІнтеграл ≈ {integral:.6f}', fontsize=14)
        ax.axvline(x=low, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=high, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return integral


def convergence_analysis(fn: Callable, low: float, high: float,
                         true_value: float, max_points: int = 100000,
                         func_name: str = "f(x)"):
    """
    Аналізує збіжність методу Монте-Карло зі збільшенням кількості точок.

    Args:
        fn: Функція для інтегрування
        low: Межі інтегрування
        high: Межі інтегрування
        true_value: Точне значення інтеграла
        max_points: Максимальна кількість точок
        func_name: Назва функції
    """
    n_values = np.unique(np.logspace(2, int(np.log10(max_points)), 50, dtype=int))
    results = []
    errors = []
    times = []
    std_devs = []

    print(f"\nАналіз збіжності методу Монте-Карло для {func_name}")
    print("-" * 70)

    for n in n_values:
        start_time = time.perf_counter()

        # Виконуємо кілька ітерацій для усереднення
        iterations = 20
        iter_results = []
        for _ in range(iterations):
            result, _, _ = monte_carlo_integration(fn, low, high, n)
            iter_results.append(result)

        avg_result = np.mean(iter_results)
        std_dev = np.std(iter_results)
        elapsed_time = time.perf_counter() - start_time

        results.append(avg_result)
        errors.append(abs(float(avg_result) - true_value))
        times.append(elapsed_time / iterations)
        std_devs.append(std_dev)

        if n in [100, 1000, 10000, 100000]:
            print(f"n = {n:7,}: Результат = {avg_result:11.6f}, "
                  f"Похибка = {abs(float(avg_result) - true_value):9.6f}, "
                  f"σ = {std_dev:9.6f}, "
                  f"Час = {times[-1] * 1000:6.2f} мс")

    # Візуалізація збіжності
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Графік результатів
    axes[0, 0].semilogx(n_values, results, 'b-', linewidth=2)
    axes[0, 0].axhline(y=true_value, color='r', linestyle='--',
                       label=f'Точне значення = {true_value:.6f}')
    axes[0, 0].fill_between(n_values,
                            np.array(results) - np.array(std_devs),
                            np.array(results) + np.array(std_devs),
                            alpha=0.3, color='blue', label='±σ')
    axes[0, 0].set_xlabel('Кількість точок')
    axes[0, 0].set_ylabel('Значення інтеграла')
    axes[0, 0].set_title('Збіжність до точного значення')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Графік похибки
    axes[0, 1].loglog(n_values, errors, 'r-', linewidth=2, label='Фактична похибка')
    axes[0, 1].loglog(n_values, 1 / np.sqrt(n_values), 'g--',
                      label='O(1/√n) - теоретична збіжність')
    axes[0, 1].set_xlabel('Кількість точок')
    axes[0, 1].set_ylabel('Абсолютна похибка')
    axes[0, 1].set_title('Зменшення похибки')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Графік стандартного відхилення
    axes[1, 0].loglog(n_values, std_devs, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Кількість точок')
    axes[1, 0].set_ylabel('Стандартне відхилення')
    axes[1, 0].set_title('Стабільність результату')
    axes[1, 0].grid(True, alpha=0.3)

    # Графік часу виконання
    axes[1, 1].loglog(n_values, times, 'g-', linewidth=2)
    axes[1, 1].loglog(n_values, n_values / n_values[0] * times[0], 'r--',
                      label='O(n) - лінійна складність')
    axes[1, 1].set_xlabel('Кількість точок')
    axes[1, 1].set_ylabel('Час виконання (с)')
    axes[1, 1].set_title('Час обчислення')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle(f'Аналіз збіжності для {func_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

    return n_values, results, errors


def compare_methods(fn: Callable, low: float, high: float, func_name: str = "f(x)"):
    """
    Порівнює різні методи обчислення інтеграла.

    Args:
        fn: Функція для інтегрування
        low: Межі інтегрування
        high: Межі інтегрування
        func_name: Назва функції для відображення
    """
    print("\n" + "=" * 70)
    print("ПОРІВНЯННЯ МЕТОДІВ ОБЧИСЛЕННЯ ІНТЕГРАЛА")
    print("=" * 70)
    print(f"Функція: {func_name}")
    print(f"Межі інтегрування: від {low} до {high}")
    print("-" * 70)

    results = {}

    # 1. Метод quad з scipy (еталонний результат)
    quad_result, quad_error = spi.quad(fn, low, high)
    results['SciPy quad'] = quad_result
    print(f"\n1. SciPy quad (еталон): {quad_result:.10f}")
    print(f"   Оцінка похибки: {quad_error:.2e}")

    # Використовуємо quad як еталонне значення
    true_value = quad_result

    # 2. Метод Монте-Карло з різною кількістю точок
    print("\n2. Метод Монте-Карло:")
    print(f"{'Точок':<10} {'Результат':<15} {'Похибка':<15} {'Відн. похибка':<15} {'Час (мс)':<10}")
    print("-" * 70)

    monte_carlo_results = []

    for n_points in [1000, 10000, 50000, 100000, 500000, 1000000]:
        # Виконуємо кілька ітерацій для отримання середнього та стандартного відхилення
        iterations = 10 if n_points <= 100000 else 5
        iter_results = []

        start_time = time.perf_counter()
        for _ in range(iterations):
            result, _, _ = monte_carlo_integration(fn, low, high, n_points)
            iter_results.append(result)
        elapsed_time = time.perf_counter() - start_time

        mean_result = np.mean(iter_results)
        std_result = np.std(iter_results)

        monte_carlo_results.append((n_points, mean_result, std_result))
        results[f'MC-{n_points}'] = mean_result

        error = abs(mean_result - true_value)
        relative_error = error / abs(true_value) * 100 if true_value != 0 else 0

        print(f"{n_points:<10,} {mean_result:<15.10f} {error:<15.10f} "
              f"{relative_error:<15.6f}% {elapsed_time / iterations * 1000:<10.2f}")

    # 3. Додаткові адаптивні методи з scipy
    print("\n3. Інші методи інтегрування:")

    # Simpson's rule
    try:
        from scipy.integrate import simpson
        x_simpson = np.linspace(low, high, 1001)
        y_simpson = fn(x_simpson)
        simpson_result = simpson(y_simpson, x=x_simpson)
        results['Simpson'] = simpson_result
        print(f"   Simpson's rule: {simpson_result:.10f}")
    except:
        pass

    # Romberg integration
    try:
        from scipy.integrate import romberg
        romberg_result = romberg(fn, low, high)
        results['Romberg'] = romberg_result
        print(f"   Romberg:        {romberg_result:.10f}")
    except:
        pass

    # Візуалізація порівняння
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Графік значень
    methods = list(results.keys())
    values = list(results.values())
    colors = ['red' if 'MC' in m else 'blue' for m in methods]

    ax1.barh(methods, values, color=colors, alpha=0.7)
    ax1.axvline(x=true_value, color='green', linestyle='--',
                label=f'SciPy quad = {true_value:.6f}')
    ax1.set_xlabel('Значення інтеграла')
    ax1.set_title('Порівняння результатів')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Графік похибок для Монте-Карло
    mc_points = [r[0] for r in monte_carlo_results]
    mc_errors = [abs(r[1] - true_value) for r in monte_carlo_results]

    ax2.loglog(mc_points, mc_errors, 'ro-', label='Фактична похибка')
    ax2.loglog(mc_points, 1 / np.sqrt(mc_points), 'g--', label='O(1/√n)')
    ax2.set_xlabel('Кількість точок')
    ax2.set_ylabel('Абсолютна похибка')
    ax2.set_title('Похибка методу Монте-Карло')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Порівняння методів для {func_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

    return results


def test_different_functions():
    """
    Тестує метод Монте-Карло на різних функціях.
    """
    print("\n" + "=" * 70)
    print("ТЕСТУВАННЯ РІЗНИХ ФУНКЦІЙ")
    print("=" * 70)

    test_cases = [
        {
            'name': 'f(x) = x²',
            'func': lambda x: x ** 2,
            'a': 0, 'b': 2,
            'analytical': 8 / 3
        },
        {
            'name': 'f(x) = e^(-x²) (Гаусова)',
            'func': lambda x: np.exp(-x ** 2),
            'a': -2, 'b': 2,
            'analytical': np.sqrt(np.pi) * sps.erf(2)
        },
        {
            'name': 'f(x) = sin(x²)·e^(-x/2)',
            'func': lambda x: np.sin(x ** 2) * np.exp(-x / 2),
            'a': 0, 'b': 4,
            'analytical': None  # Немає простої аналітичної форми
        },
        {
            'name': 'f(x) = x·sin(x)·cos(2x)',
            'func': lambda x: x * np.sin(x) * np.cos(2 * x),
            'a': 0, 'b': np.pi,
            'analytical': None
        },
        {
            'name': 'f(x) = 1/(1+x⁴)',
            'func': lambda x: 1 / (1 + x ** 4),
            'a': 0, 'b': 2,
            'analytical': None
        }
    ]

    results_summary = []

    for test in test_cases:
        print(f"\n{test['name']}:")
        print("-" * 40)

        # SciPy quad для еталону
        quad_result, _ = spi.quad(test['func'], test['a'], test['b'])

        # Монте-Карло з 100,000 точок
        mc_results = []
        for _ in range(10):
            result, _, _ = monte_carlo_integration(test['func'], test['a'], test['b'], 100000)
            mc_results.append(result)
        mc_mean = np.mean(mc_results)
        mc_std = np.std(mc_results)

        print(f"  Межі: [{test['a']}, {test['b']}]")
        if test['analytical']:
            print(f"  Аналітичне: {test['analytical']:.6f}")
        print(f"  SciPy quad: {quad_result:.6f}")
        print(f"  Монте-Карло (100k): {mc_mean:.6f} ± {mc_std:.6f}")
        print(f"  Похибка: {abs(mc_mean - quad_result):.6f}")

        results_summary.append({
            'function': test['name'],
            'quad': quad_result,
            'monte_carlo': mc_mean,
            'error': abs(mc_mean - quad_result)
        })

    return results_summary


def main():
    """
    Головна функція для виконання всіх обчислень та візуалізацій.
    """

    print("=" * 70)
    print("ОБЧИСЛЕННЯ ІНТЕГРАЛА МЕТОДОМ МОНТЕ-КАРЛО")
    print("=" * 70)

    # Визначаємо складну функцію для демонстрації
    def f(x):
        """Складна композитна функція: sin(x²)·e^(-x/2)"""
        return np.sin(x ** 2) * np.exp(-x / 2)

    func_name = "f(x) = sin(x²)·e^(-x/2)"
    a = 0  # Нижня межа
    b = 4  # Верхня межа

    print(f"\nОбрана функція: {func_name}")
    print(f"Межі інтегрування: від {a} до {b}")
    print("\nЦя функція не має простого аналітичного розв'язку,")
    print("що робить її ідеальною для демонстрації методу Монте-Карло!")

    # Обчислюємо еталонне значення через SciPy
    true_value, error = spi.quad(f, a, b)
    print(f"\nЕталонне значення (SciPy quad): {true_value:.10f}")

    # 1. Візуалізація функції та області інтегрування
    print("\n1. ВІЗУАЛІЗАЦІЯ ФУНКЦІЇ")
    print("-" * 70)

    x = np.linspace(a - 0.5, b + 0.5, 1000)
    y = f(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Основний графік
    ax1.plot(x, y, 'b-', linewidth=2.5, label=func_name)
    ix = np.linspace(a, b, 500)
    iy = f(ix)
    ax1.fill_between(ix, iy, color='skyblue', alpha=0.3, label='Область інтегрування')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axvline(x=a, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=b, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlim([a - 0.5, b + 0.5])
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title(f'Графік функції {func_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Детальний вигляд осциляцій
    x_detail = np.linspace(0, 4, 1000)
    y_detail = f(x_detail)
    ax2.plot(x_detail, y_detail, 'r-', linewidth=1.5)
    ax2.fill_between(x_detail, y_detail, color='salmon', alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Детальний вигляд осциляцій')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)

    # Додаємо текст з інформацією
    info_text = f'Інтеграл ≈ {true_value:.6f}\nФункція має:\n• Швидкі осциляції (sin(x²))\n• Експоненційне затухання (e^(-x/2))'
    ax1.text(2.5, 0.3, info_text,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.show()

    # 2. Демонстрація методу Монте-Карло з візуалізацією точок
    print("\n2. ВІЗУАЛІЗАЦІЯ МЕТОДУ МОНТЕ-КАРЛО")
    print("-" * 70)

    for n_points in [1000, 5000, 20000]:
        print(f"\nОбчислення з {n_points:,} точками:")
        result = monte_carlo_with_visualization(f, a, b,
                                                n_points=n_points,
                                                show_points=True,
                                                func_name=func_name)
        print(f"  Результат: {result:.6f}")
        print(f"  Похибка: {abs(result - true_value):.6f}")

    # 3. Порівняння методів
    print("\n3. ПОРІВНЯННЯ МЕТОДІВ")
    results = compare_methods(f, a, b, func_name)

    # 4. Аналіз збіжності
    print("\n4. АНАЛІЗ ЗБІЖНОСТІ")
    convergence_analysis(f, a, b, true_value, max_points=200000, func_name=func_name)

    # 5. Тестування на різних функціях
    print("\n5. ТЕСТУВАННЯ НА РІЗНИХ ФУНКЦІЯХ")
    test_different_functions()


if __name__ == "__main__":
    main()