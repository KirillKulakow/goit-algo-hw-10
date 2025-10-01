import time
import matplotlib.pyplot as plt
from . import find_coins_greedy, find_min_coins

def measure_time(func, amount, coins, iterations=100) -> float:
    """
    Вимірює час виконання функції.

    Args:
        func: Функція для вимірювання
        amount: Сума для тестування
        coins: Список монет
        iterations: Кількість ітерацій для усереднення

    Returns:
        Середній час виконання в секундах
    """
    total_time = 0
    for _ in range(iterations):
        start = time.perf_counter()
        func(amount, coins)
        end = time.perf_counter()
        total_time += (end - start)

    return total_time / iterations


def compare_algorithms(max_amount=1000, step=50) -> dict:
    """
    Порівнює ефективність двох алгоритмів.

    Args:
        max_amount: Максимальна сума для тестування
        step: Крок збільшення суми

    Returns:
        Словник з результатами порівняння
    """
    coins = [50, 25, 10, 5, 2, 1]
    amounts = list(range(step, max_amount + 1, step))

    greedy_times = []
    dp_times = []

    print("Порівняння алгоритмів...")
    print("-" * 60)

    for amount in amounts:
        greedy_time = measure_time(find_coins_greedy, amount, coins)
        dp_time = measure_time(find_min_coins, amount, coins)

        greedy_times.append(greedy_time * 1000)  # Конвертуємо в мілісекунди
        dp_times.append(dp_time * 1000)

        if amount % 200 == 0:  # Виводимо прогрес кожні 200 одиниць
            print(f"Сума {amount}: Жадібний = {greedy_time * 1000:.4f} мс, ДП = {dp_time * 1000:.4f} мс")

    return {
        'amounts': amounts,
        'greedy_times': greedy_times,
        'dp_times': dp_times
    }


def visualize_comparison(results):
    """
    Візуалізує порівняння алгоритмів.

    Args:
        results: Словник з результатами порівняння
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results['amounts'], results['greedy_times'], label='Жадібний алгоритм', marker='o')
    plt.plot(results['amounts'], results['dp_times'], label='Динамічне програмування', marker='s')
    plt.xlabel('Сума')
    plt.ylabel('Час виконання (мс)')
    plt.title('Порівняння часу виконання алгоритмів')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    ratio = [dp / greedy for greedy, dp in zip(results['greedy_times'], results['dp_times'])]
    plt.plot(results['amounts'], ratio, color='red', marker='^')
    plt.xlabel('Сума')
    plt.ylabel('Співвідношення ДП/Жадібний')
    plt.title('Відносна швидкість (ДП / Жадібний)')
    plt.axhline(y=1, color='green', linestyle='--', label='Однакова швидкість')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def verify_correctness(amount=113):
    """
    Перевіряє коректність роботи алгоритмів.

    Args:
        amount: Сума для перевірки
    """
    coins = [50, 25, 10, 5, 2, 1]

    print(f"\nПеревірка коректності для суми {amount}:")
    print("-" * 60)

    # Жадібний алгоритм
    greedy_result = find_coins_greedy(amount, coins)
    greedy_count = sum(greedy_result.values())
    greedy_sum = sum(coin * count for coin, count in greedy_result.items())

    print(f"Жадібний алгоритм:")
    print(f"  Результат: {greedy_result}")
    print(f"  Кількість монет: {greedy_count}")
    print(
        f"  Перевірка суми: {greedy_sum} = {amount} ✓" if greedy_sum == amount else f"  Помилка: {greedy_sum} ≠ {amount}")

    # Динамічне програмування
    dp_result = find_min_coins(amount, coins)
    dp_count = sum(dp_result.values())
    dp_sum = sum(coin * count for coin, count in dp_result.items())

    print(f"\nДинамічне програмування:")
    print(f"  Результат: {dp_result}")
    print(f"  Кількість монет: {dp_count}")
    print(f"  Перевірка суми: {dp_sum} = {amount} ✓" if dp_sum == amount else f"  Помилка: {dp_sum} ≠ {amount}")

    # Порівняння
    print(f"\nПорівняння:")
    if greedy_count == dp_count:
        print(f"  Обидва алгоритми використовують {greedy_count} монет(и) ✓")
    else:
        print(f"  Жадібний: {greedy_count} монет(и), ДП: {dp_count} монет(и)")
        print(f"  ДП ефективніший на {greedy_count - dp_count} монет(и)")


def test_edge_cases():
    """
    Тестує граничні випадки.
    """
    coins = [50, 25, 10, 5, 2, 1]
    test_cases = [0, 1, 7, 30, 99, 113, 250, 999]

    print("\nТестування граничних випадків:")
    print("-" * 60)

    for amount in test_cases:
        greedy = find_coins_greedy(amount, coins)
        dp = find_min_coins(amount, coins)

        greedy_count = sum(greedy.values())
        dp_count = sum(dp.values())

        status = "✓" if greedy_count == dp_count else f"ДП краще на {greedy_count - dp_count}"
        print(f"Сума {amount:3}: Жадібний = {greedy_count:2} монет, ДП = {dp_count:2} монет | {status}")
