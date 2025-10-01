def find_coins_greedy(amount, coins=None) -> dict:
    """
    Жадібний алгоритм для визначення решти.

    Args:
        amount: Сума для видачі
        coins: Список номіналів монет (відсортований за спаданням)

    Returns:
        Словник з кількістю монет кожного номіналу
    """
    if coins is None:
        coins = [50, 25, 10, 5, 2, 1]
    result = {}
    remaining = amount

    for coin in coins:
        if remaining >= coin:
            count = remaining // coin
            result[coin] = count
            remaining -= coin * count

        if remaining == 0:
            break

    return result