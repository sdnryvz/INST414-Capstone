# pulling local data separate from API

import pandas as pd 
import numpy as np 
import re 
from unidecode import unidecode 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick


pd.options.display.float_format = '{:.0f}'.format

# Reading the dataset
df1 = pd.read_csv("/Users/nuryavuz/Desktop/inst414-capstone/tourism_arrivals.csv")
df1.head()

# Counting number of NA values dropped 
before = len(df1)
df1 = df1.dropna()
after = len(df1)

before
after

# -------------------------------
# Extracting Country from Title 
# -------------------------------

yt_df = pd.read_csv("/Users/nuryavuz/Desktop/inst414-capstone/travel_vlogs_2017_2019-1.csv")

cities = pd.read_csv("/Users/nuryavuz/Desktop/inst414-capstone/worldcities.csv")
cities = cities[["city", "country"]]

cities['city'] = cities['city'].apply(lambda x: unidecode(str(x).lower()))
cities['country'] = cities['country'].apply(lambda x: unidecode(str(x).lower()))
yt_df['Title'] = yt_df['Title'].apply(lambda x: unidecode(str(x).lower()))


city_to_country = {row['city']: row['country'] for _, row in cities.iterrows()}

country_list = sorted(cities['country'].unique(), key=len, reverse=True)
city_list = sorted(city_to_country.keys(), key=len, reverse=True)

location_pattern = r'\b(' + '|'.join(re.escape(name) for name in (city_list + country_list)) + r')\b'


def extract_countries(title):
    matches = re.findall(location_pattern, title)
    if not matches:
        return None

    countries = set()
    for name in matches:
        # If the match is a city, map it to its country
        if name in city_to_country:
            countries.add(city_to_country[name].title())
        else:
            countries.add(name.title())

    # Return sorted unique list (e.g., "France, Spain")
    return ", ".join(sorted(countries))

yt_df['Country'] = yt_df['Title'].apply(extract_countries)

yt_df.to_csv("/Users/nuryavuz/desktop/vlogs_with_country_fixed.csv", index=False)


yt_df = pd.read_csv("/Users/nuryavuz/desktop/vlogs_with_country_fixed.csv")
clean_df = yt_df.dropna()
len(clean_df)
clean_df.describe()
clean_df.shape

# -------------------------------
# Identify Outliers in YT data
# -------------------------------

print(clean_df.select_dtypes(include=[np.number]).columns)
print(clean_df.head())

clean_df['View Count'] = pd.to_numeric(clean_df['View Count'], errors='coerce')


Q1 = clean_df['View Count'].quantile(0.25)
Q3 = clean_df['View Count'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"IQR Range: {lower:.0f}  →  {upper:.0f}")

# Mark outliers
clean_df.loc[:, 'is_outlier'] = (clean_df['View Count'] < lower) | (clean_df['View Count'] > upper)

# Count outliers
num_outliers = clean_df['is_outlier'].sum()
total_videos = len(clean_df)

print(f"🚨 Outliers detected: {num_outliers} out of {total_videos} videos "
      f"({num_outliers/total_videos*100:.1f}% of dataset)")

# Remove outliers
clean_no_outliers = clean_df[~clean_df['is_outlier']].copy()
print(f"✅ Cleaned dataset size: {len(clean_no_outliers)} videos remain")

clean_no_outliers = clean_no_outliers[clean_no_outliers['View Count'] >= 0]


# ----------------------------------
# Boxplot visualization
# ----------------------------------
Q1 = clean_no_outliers['View Count'].quantile(0.25)
Q3 = clean_no_outliers['View Count'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

plt.figure(figsize=(10, 4))
ax = sns.boxplot(
    x='View Count',
    data=clean_no_outliers,
    color="#9ecae1",       
    width=0.4,
    showfliers=True        
)
lb = plt.axvline(lower, color='red', linestyle='--', linewidth=2)
ub = plt.axvline(upper, color='red', linestyle='--', linewidth=2)

plt.title("Boxplot of YouTube View Counts", fontsize=14)
plt.xlabel("View Count (number of views per video, tens of thousands)", fontsize=12)

legend_elements = [
    Patch(facecolor="#9ecae1", edgecolor='black', label="Box = IQR (Q1–Q3)"),
    Line2D([0], [0], color='black', linewidth=2, label="Median line (inside box)"),
    Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label="Whiskers (non-outlier range)"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, linestyle='None', label="Fliers = Outliers"),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2, label="IQR outlier thresholds")
]

plt.legend(handles=legend_elements, title="Key", loc="upper right", frameon=True)

ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))
ax.set_xlabel("View Count (thousands of views)")

plt.tight_layout()
plt.show()

# ----------------------------------
# Histogram visualization
# ----------------------------------

plt.figure(figsize=(10, 5))
sns.histplot(clean_no_outliers['View Count'], bins=50, kde=True, color="skyblue")

plt.axvline(lower, color='red', linestyle='--', linewidth=2, label='Lower Outlier Bound')
plt.axvline(upper, color='red', linestyle='--', linewidth=2, label='Upper Outlier Bound')

plt.title("📊 Distribution of YouTube Video View Counts", fontsize=14)
plt.xlabel("View Count (Number of Views per Video, Thousands)", fontsize=12)
plt.ylabel("Frequency (Number of Videos, Hundreds)", fontsize=12)

plt.text(upper * 0.8, plt.ylim()[1] * 0.8, 
         "🔹 Each bar = number of videos\nin that view range\n🔴 Red lines = outlier thresholds",
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))



ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
ax.set_xlabel("View Count (thousands of views)")

plt.legend()
plt.tight_layout()
plt.show()


clean_no_outliers.describe()

##############################

for col in ['View Count', 'Likes', 'Comments', 'Duration (sec)']:
    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

# Drop missing rows for core variables
final_df = clean_df.dropna(subset=['View Count', 'Likes', 'Comments', 'Duration (sec)'])


# Part 3A histogram and density plot 



numeric_vars = ['View Count', 'Likes', 'Comments', 'Duration (sec)']

plt.figure(figsize=(14, 8))
for i, col in enumerate(numeric_vars, 1):
    plt.subplot(2, 2, i)
    sns.histplot(final_df[col], bins=40, kde=True, color='skyblue')
    plt.title(f"Distribution of {col}", fontsize=12)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
plt.show()

summary = clean_df[numeric_vars].describe().T
summary['variance'] = final_df[numeric_vars].var()
summary['skewness'] = final_df[numeric_vars].skew()
summary['kurtosis'] = final_df[numeric_vars].kurt()
print("📊 Summary Statistics:")
print(summary.round(2))

# Part 3B scatterplots and relationships

final_df = pd.read_csv("/Users/nuryavuz/Desktop/inst414-capstone/travel_vlogs_2017_2019_v2.csv")
numeric_vars = ['View Count', 'Likes', 'Comments']
for col in numeric_vars:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

sns.set(style="whitegrid")
pairs = [
    ('View Count', 'Likes'),
    ('View Count', 'Comments')
]

plt.figure(figsize=(14, 4))
for i, (x, y) in enumerate(pairs, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(data=final_df, x=x, y=y, alpha=0.6, color='teal')
    sns.regplot(data=final_df, x=x, y=y, scatter=False, color='red')
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
plt.tight_layout()
plt.show()

#correlation matrix
plt.figure(figsize=(6, 4))
corr = final_df[numeric_vars].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Engagement Metrics")
plt.show()

#grouped comparison by country

top_countries = clean_no_outliers['Country'].value_counts().head(8).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=clean_no_outliers[clean_no_outliers['Country'].isin(top_countries)],
            x='Country', y='View Count', palette='Set3')
plt.title("Distribution of View Counts by Country")
plt.xlabel("Country")
plt.ylabel("View Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

## C

plt.figure(figsize=(8,5))
sns.scatterplot(data=final_df, x='Duration (sec)', y='View Count', alpha=0.6, color='royalblue')
sns.regplot(data=final_df, x='Duration (sec)', y='View Count', scatter=False, color='red')

plt.title("📺 Relationship Between Video Duration and View Count", fontsize=14)
plt.xlabel("Duration (Seconds)", fontsize=12)
plt.ylabel("View Count", fontsize=12)
plt.legend(labels=["Trend line (linear regression)"], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



#################

# final 4 
plt.figure(figsize=(12, 6))
top_countries = clean_no_outliers['Country'].value_counts().head(8).index
sns.boxplot(
    data=clean_no_outliers[clean_no_outliers['Country'].isin(top_countries)],
    x='Country', 
    y='View Count', 
    palette='Set3'
)

plt.title("Distribution of YouTube View Counts by Country", fontsize=15, fontweight='bold')
plt.xlabel("Country", fontsize=12)
plt.ylabel("View Count (log scale for readability)", fontsize=12)
plt.yscale('log')  # optional if data is skewed
plt.xticks(rotation=30, fontsize=11)
plt.tight_layout()
plt.show()

print("📝 Caption: Countries such as Japan and Italy show higher median view counts, suggesting stronger travel vlog engagement in those regions.")

plt.figure(figsize=(10, 5))
sns.histplot(clean_no_outliers['View Count'], bins=40, kde=True, color="skyblue")
plt.title("Distribution of YouTube Video View Counts", fontsize=15, fontweight='bold')
plt.xlabel("View Count (thousands of views)", fontsize=12)
plt.ylabel("Number of Videos", fontsize=12)
plt.tight_layout()
plt.show()

print("📝 Caption: Most travel vlogs receive fewer than 100,000 views, but a few viral videos heavily skew the distribution.")

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=final_df,
    x='View Count', 
    y='Likes', 
    alpha=0.6, 
    color='teal', 
    label='Video'
)
sns.regplot(
    data=final_df, 
    x='View Count', 
    y='Likes', 
    scatter=False, 
    color='red', 
    label='Trend Line'
)

plt.title("Relationship Between Views and Likes", fontsize=15, fontweight='bold')
plt.xlabel("View Count (log scale)", fontsize=12)
plt.ylabel("Likes", fontsize=12)
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.show()

print("📝 Caption: Likes increase consistently with view count, confirming engagement scales roughly linearly with audience size.")

plt.figure(figsize=(6, 4))
corr = final_df[['View Count', 'Likes', 'Comments']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title("Correlation Matrix: YouTube Engagement Metrics", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

print("📝 Caption: High positive correlations suggest that as view counts increase, both likes and comments follow — reinforcing engagement consistency across metrics.")


from scipy.stats import ttest_ind

india = clean_no_outliers[clean_no_outliers['Country'] == 'India']['View Count']
usa = clean_no_outliers[clean_no_outliers['Country'] == 'United States']['View Count']

t_stat, p_val = ttest_ind(india, usa, equal_var=False, nan_policy='omit')
print(f"T-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")
