# run_cli.py
"""
Command Line Interface for Smart Ingredient Watchdog
"""

from watchdog_core import IngredientWatchdog

def main():
    print("=" * 70)
    print("🕵️ SMART INGREDIENT WATCHDOG - CLI MODE")
    print("=" * 70)

    watchdog = IngredientWatchdog()
    success = watchdog.run_complete_system()

    if not success:
        print("❌ Failed to initialize system. Check dataset/model.")
        return

    print("\n🎉 System initialized successfully!")
    print("Type a list of ingredients (comma separated), or type 'exit' to quit.\n")

    while True:
        user_input = input("👉 Enter ingredients: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Exiting Smart Ingredient Watchdog. Stay safe!")
            break

        ingredients = watchdog.parse_ingredients_from_text(user_input)
        results = watchdog.analyze_ingredients(ingredients)

        print("\n🔍 Analysis Result:")
        print("- Safe:", [item["name"] for item in results["safe"]])
        print("- Harmful:")
        for item in results["harmful"]:
            print(f"   ⚠ {item['name']} ({item['side_effects']} | Risk: {item['risk']})")

        print("- Unknown:")
        for item in results["unknown"]:
            print(f"   ❓ {item['name']} (no database info)")


if __name__ == "__main__":
    main()
