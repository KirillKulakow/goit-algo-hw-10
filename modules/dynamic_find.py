from collections import defaultdict

def find_min_coins(amount, coins=None) -> dict:
    """
    Алгоритм динамічного програмування для знаходження мінімальної кількості монет.

    Args:
        amount: Сума для видачі
        coins: Список номіналів монет

    Returns:
        Словник з кількістю монет кожного номіналу
    """
    if coins is None:
        coins = [50, 25, 10, 5, 2, 1]
    # dp[i] - мінімальна кількість монет для суми i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    # parent[i] - номінал монети, використаної для досягнення суми i
    parent = [-1] * (amount + 1)

    # Заповнюємо таблицю динамічного програмування
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin

    # Якщо неможливо видати суму
    if dp[amount] == float('inf'):
        return {}

    # Відновлюємо розв'язок
    result = defaultdict(int)
    current = amount
    while current > 0:
        coin = parent[current]
        result[coin] += 1
        current -= coin

    # Конвертуємо в звичайний словник та сортуємо
    return dict(sorted(result.items(), reverse=True))