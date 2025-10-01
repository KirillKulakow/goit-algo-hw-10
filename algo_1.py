from modules.testing import verify_correctness, test_edge_cases, compare_algorithms, visualize_comparison


def main():
    """
    Головна функція для запуску всіх тестів та порівнянь.
    """
    print("=" * 60)
    print("СИСТЕМА ВИДАЧІ РЕШТИ: ПОРІВНЯННЯ АЛГОРИТМІВ")
    print("=" * 60)

    # 1. Перевірка коректності
    verify_correctness(113)

    # 2. Тестування граничних випадків
    test_edge_cases()

    # 3. Порівняння продуктивності
    print("\n" + "=" * 60)
    print("АНАЛІЗ ПРОДУКТИВНОСТІ")
    print("=" * 60)

    results = compare_algorithms(max_amount=2000, step=100)

    # 4. Візуалізація результатів
    visualize_comparison(results)

    # 5. Висновки
    avg_greedy = sum(results['greedy_times']) / len(results['greedy_times'])
    avg_dp = sum(results['dp_times']) / len(results['dp_times'])

    print("\n" + "=" * 60)
    print("ПІДСУМКОВІ РЕЗУЛЬТАТИ")
    print("=" * 60)
    print(f"Середній час виконання:")
    print(f"  Жадібний алгоритм: {avg_greedy:.4f} мс")
    print(f"  Динамічне програмування: {avg_dp:.4f} мс")
    print(f"  Співвідношення: ДП повільніший у {avg_dp / avg_greedy:.2f} разів")

    print("\nСкладність алгоритмів:")
    print("  Жадібний: O(n), де n - кількість номіналів")
    print("  ДП: O(m×n), де m - сума, n - кількість номіналів")

if __name__ == "__main__":
    main()