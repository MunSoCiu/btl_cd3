def check_data(users, books, history):
    print("\n===== DATA INFO =====")

    print("\n👤 USERS:")
    print(users.info())
    print(users.isnull().sum())

    print("\n📚 BOOKS:")
    print(books.info())
    print(books.isnull().sum())

    print("\n📖 HISTORY:")
    print(history.info())
    print(history.isnull().sum())

    print("\n===== SAMPLE =====")
    print(users.head())
    print(books.head())
    print(history.head())