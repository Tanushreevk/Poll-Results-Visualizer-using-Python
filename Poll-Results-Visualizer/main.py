import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folders if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ----------------------------
# 1. CREATE SYNTHETIC DATA
# ----------------------------
np.random.seed(42)

n = 500

data = pd.DataFrame({
    "Respondent_ID": range(1, n+1),
    "Region": np.random.choice(["North", "South", "East", "West"], n),
    "Age_Group": np.random.choice(["18-25", "26-35", "36-50"], n),
    "Option": np.random.choice(["Product A", "Product B", "Product C"], n),
    "Date": pd.date_range(start="2024-01-01", periods=n, freq="D")
})

# Save dataset
data.to_csv("data/poll_data.csv", index=False)

print("\n✅ Dataset Created:\n")
print(data.head())

# ----------------------------
# 2. CLEAN DATA
# ----------------------------
data.dropna(inplace=True)

# ----------------------------
# 3. ANALYSIS
# ----------------------------

vote_counts = data["Option"].value_counts()
vote_percentage = (vote_counts / len(data)) * 100

print("\n📊 Vote Counts:\n", vote_counts)
print("\n📈 Vote Percentage:\n", vote_percentage)

# Region-wise analysis
region_analysis = pd.crosstab(data["Region"], data["Option"])
print("\n🌍 Region-wise Analysis:\n", region_analysis)

# ----------------------------
# 3B. AGE GROUP ANALYSIS
# ----------------------------

age_analysis = pd.crosstab(data["Age_Group"], data["Option"])
print("\n👥 Age-wise Analysis:\n", age_analysis)

# ----------------------------
# 3C. TREND ANALYSIS (DATE)
# ----------------------------

trend = data.groupby(["Date", "Option"]).size().unstack().fillna(0)

print("\n📅 Trend Analysis:\n", trend.head())
# ----------------------------
# 4. VISUALIZATION
# ----------------------------

# Bar Chart
plt.figure()
sns.barplot(x=vote_counts.index, y=vote_counts.values)
plt.title("Overall Poll Results")
plt.savefig("images/bar_chart.png")
plt.close()

# Pie Chart
plt.figure()
plt.pie(vote_counts, labels=vote_counts.index, autopct='%1.1f%%')
plt.title("Vote Share")
plt.savefig("images/pie_chart.png")
plt.close()

# Region-wise Stacked Chart
plt.figure()
region_analysis.plot(kind='bar', stacked=True)
plt.title("Region-wise Poll Results")
plt.savefig("images/region_chart.png")
plt.close()

# Age-wise chart
plt.figure()
age_analysis.plot(kind='bar', stacked=True)
plt.title("Age-wise Poll Results")
plt.savefig("images/age_chart.png")
plt.close()

# Trend chart
plt.figure()
trend.plot()
plt.title("Poll Trends Over Time")
plt.savefig("images/trend_chart.png")
plt.close()

# ----------------------------
# 5. INSIGHTS
# ----------------------------

top_option = vote_counts.idxmax()
print(f"\n🏆 Leading Choice: {top_option}")
vote_counts.to_csv("outputs/vote_counts.csv")
region_analysis.to_csv("outputs/region_analysis.csv")
age_analysis.to_csv("outputs/age_analysis.csv")