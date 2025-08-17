import streamlit as st
from watchdog_core import IngredientWatchdog

# Initialize system
st.set_page_config(page_title="Smart Ingredient Watchdog", page_icon="🕵️", layout="centered")
st.title("🕵️ Smart Ingredient Watchdog")
st.write("AI system to detect **harmful food additives** from ingredients.")

# Initialize Watchdog
watchdog = IngredientWatchdog()
success = watchdog.run_complete_system()

if not success:
    st.error("❌ Failed to initialize system. Please check dataset/model.")
else:
    st.success("✅ System initialized successfully!")

    # ---------------- Input Section ----------------
    st.subheader("Enter Ingredients")
    user_input = st.text_area("Type ingredients (comma separated):", 
                              "Water, Sugar, Sodium Benzoate, MSG")

    if st.button("Analyze"):
        ingredients = watchdog.parse_ingredients_from_text(user_input)
        results = watchdog.analyze_ingredients(ingredients)

        # ---------------- Results Section ----------------
        st.subheader("🔍 Analysis Result")

        if results["safe"]:
            st.success("### ✅ Safe Ingredients")
            for item in results["safe"]:
                st.write(f"- **{item['name']}** → {item.get('info','Safe')}")

        if results["harmful"]:
            st.error("### ⚠ Harmful Ingredients")
            for item in results["harmful"]:
                st.write(f"- **{item['name']}** → {item['side_effects']} (Risk: {item['risk']})")

        if results["unknown"]:
            st.warning("### ❓ Unknown Ingredients")
            for item in results["unknown"]:
                st.write(f"- **{item['name']}** → Not in database / AI unsure")
