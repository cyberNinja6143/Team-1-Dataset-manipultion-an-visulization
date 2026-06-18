#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Loading the data

# In[4]:


df = pd.read_csv('24_Coffee_Taste_Test.csv')
df.head()


# # Raw data overview
# ## Shape and Data types
# The dataset has 4042 rows and 57 columns of which 13 are floats and 44 are objects.

# In[5]:


df.shape


# In[6]:


df.info()


# ## Null Counts

# In[7]:


df.isnull().sum().sort_values()


# ## Basic Statistics

# In[8]:


float_cols = df.select_dtypes(include=["float"]).columns.tolist()
df[float_cols].describe()


# ## Interesting columns
# ### favorite
# This column denotes the subject's favorite type of coffee 

# In[9]:


df.favorite.value_counts(dropna=False)


# ### gender
# This column denotes the subject's gender

# In[10]:


df.gender.value_counts(dropna=False)


# ### total_spend
# This column denotes the range of money the subject spends on coffee per month

# In[11]:


df.total_spend.value_counts(dropna=False)


# ## age
# This column denotes the age bracket of the subject

# In[12]:


df.age.value_counts(dropna=False)


# ### wfh
# This column denotes whether the subject works from home / office.

# In[13]:


df.wfh.value_counts(dropna=False)


# This column has a lot of missing values, but would be interesting to analyze with the column `where_drink`

# ### spent_equipment
# This column denotes how much a subject has spent on equipment to brew coffee

# In[14]:


df.spent_equipment.value_counts(dropna=False)


# ### cups
# This column denotes the average cups of a coffee a subject drinks in a day.

# In[15]:


df.cups.value_counts(dropna=False)


# In[16]:


df.employment_status.value_counts(dropna=False)


# To conclude, the data has a lot of missing values, but it does not suffer from inconsistent formatting, outliers or duplicate values. So the only cleaning required is deciding what to do with the missing values

# # Cleaning the data

# ### Remove malicious code
# Prevents cross site scripting by removing html tags.

# In[17]:


for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.replace(r"<[^>]*>|(?i)javascript:", "", regex=True)


# ### favorite
# This column has only 62 missing values and over 10 categories.<br/> 
# It would not make sense to impute a value here, since we cannot be certain what a subject's favorite coffee is.<br/>
# Because there are very few missing values, we can safely drop them.

# In[18]:


df = df.dropna(subset=["favorite"])
df.favorite.value_counts(dropna=False)


# ### gender
# This column has over 500 missing values.<br/>
# However, we can reasonably assume that the people who did not specify a gender, preferred not to share their gender. <br/>
# So we fill the nan values in this column with `Prefer not to say`

# In[19]:


df['gender'] = df['gender'].fillna('Prefer not to say')
df.gender.value_counts(dropna=False)


# ### total_spend
# Dropping the missing values, since we cannot reliably estimate how much people spend on coffee monthly

# In[20]:


df = df.dropna(subset=["total_spend"])
df.total_spend.value_counts(dropna=False)


# ### age
# This column has very few missing values, and there is no reliable way to ascertain the age group for a subject, so we drop them.

# In[21]:


df = df.dropna(subset=["age"])
df.age.value_counts(dropna=False)


# ## Deriving a new column `generation`
# The column will denote the generation of a subject (Millenial, GenZ, etc..)<br/>
# The column will be derived from the age ranges in the `age` column. It is important to note that the ranges in `age` and actual generation age ranges are slightly different so the generation demographic information will be a rough estimate.

# In[22]:


generation_mapping = {
    "<18 years old": "Gen Z",
    "18-24 years old": "Gen Z",
    "25-34 years old": "Millennial",  
    "35-44 years old": "Millennial",
    "45-54 years old": "Gen X",
    "55-64 years old": "Boomer",  
    ">65 years old": "Boomer",
}

df['generation'] = df["age"].map(generation_mapping)
df['generation'].value_counts()


# ### wfh
# The category names are very long, so we shorten them to `Work from home`, `Work from office` and `Hybrid`

# In[23]:


wfh_mapping = {
    "I primarily work from home": "Work from home",
    "I primarily work in person": "Work from office",
    "I do a mix of both": "Hybrid",
}

# Force Pandas to modify the column safely in place
df.loc[:, "wfh"] = df["wfh"].map(wfh_mapping).fillna(df["wfh"])


# In[24]:


df.wfh.value_counts(dropna=False)


# This column has a significant amount of missing values.<br/>
# We can impute some values using the `employment_status` column

# In[25]:


df.employment_status.value_counts(dropna=False)


# We can assume that people who have `employment_status` value as `Unemployed`, `Retired`, `Student` do not work so we can create a new status for them as `Unemployed`.<br/>
# People with the `Homemaker` status are assumed to be working from home.

# In[26]:


# 1. Fill Unemployed, Retired, and Students as 'Unemployed'
df.loc[
    df["wfh"].isna()
    & df["employment_status"].isin(["Unemployed", "Retired", "Student"]),
    "wfh",
] = "Unemployed"

# 2. Fill Homemakers as 'Work from home'
df.loc[df["wfh"].isna() & (df["employment_status"] == "Homemaker"), "wfh"] = (
    "Work from home"
)

df.wfh.value_counts(dropna=False)


# In[27]:


df = df.dropna(subset=["wfh"])
df.wfh.value_counts(dropna=False)


# ### spent_equipment
# Dropping the missing values, since we cannot reliably estimate how much people spend on coffee equipment

# In[28]:


df = df.dropna(subset=["spent_equipment"])
df.spent_equipment.value_counts(dropna=False)


# ### cups
# Dropping the missing values, since we cannot reliably estimate how much coffee people drink per day

# In[29]:


df = df.dropna(subset=["cups"])
df.cups.value_counts(dropna=False)


# In[32]:


new_df = df[['favorite', 'gender', 'total_spend', 'age', 'wfh', 'spent_equipment', 'cups', 'generation']]
new_df.head()


# In[33]:


new_df.shape


# # Visualizations 
# 

# Here, we compare how working conditions affect the amount of money people spend on equipment to brew coffee at home and cross compare it to how much they typically spend on buying coffee monthly.

# In[34]:


# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
spending_order = [
    "Less than $20",
    "$20-$50",
    "$50-$100",
    "$100-$300",
    "$300-$500",
    "$500-$1000",
    "More than $1,000",
]

wfh_order = ["Work from office", "Hybrid", "Work from home", "Unemployed"]

# Since there is a huge gap between category counts, we normalize data across the row
# This makes data proportional (between 0% - 100%) across the rows, reducing skew
heatmap_data = (
    pd.crosstab(new_df["wfh"], new_df["spent_equipment"], normalize="index") * 100
)

heatmap_data = (
    heatmap_data.reindex(index=wfh_order, columns=spending_order).fillna(0).to_numpy()
)
heatmap_data = np.round(heatmap_data, decimals=2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
im = ax[0].imshow(heatmap_data, cmap="coolwarm")
# Show all ticks and label them with the respective list entries
ax[0].set_xticks(range(len(spending_order)), labels=spending_order, rotation=45)
ax[0].set_yticks(range(len(wfh_order)), labels=wfh_order)

# Create colorbar
#cbar = ax[1].figure.colorbar(im, ax=ax)
#cbar.ax.set_ylabel("", rotation=-90, va="bottom")

#Loop over data dimensions and create text annotations.
for i in range(len(wfh_order)):
    for j in range(len(spending_order)):
        text = ax[0].text(j, i, heatmap_data[i, j],
                       ha="center", va="center", color="w")

ax[0].set_title("Coffee Equipment Spending Habits by Working Condition")
ax[0].set_xlabel('Amount spent on Coffee Equipment')
ax[0].set_ylabel('Working conditions')

# Second heatmap
total_spend_order = [
    "<$20",
    "$20-$40",
    "$40-$60",
    "$60-$80",
    "$80-$100",
    ">$100"
]

# Since there is a huge gap between category counts, we normalize data across the row
# This makes data proportional (between 0% - 100%) across the rows, reducing skew
other_heatmap_data = (
    pd.crosstab(new_df["wfh"], new_df["total_spend"], normalize="index") * 100
)

other_heatmap_data = other_heatmap_data.reindex(
    index=wfh_order, columns=total_spend_order
).fillna(0).to_numpy()
other_heatmap_data = np.round(other_heatmap_data, decimals=2)

im = ax[1].imshow(other_heatmap_data, cmap="coolwarm")
# Show all ticks and label them with the respective list entries
ax[1].set_xticks(range(len(total_spend_order)), labels=total_spend_order, rotation=45)
ax[1].set_yticks(range(len(wfh_order)), labels=wfh_order)

# Loop over data dimensions and create text annotations.
for i in range(len(wfh_order)):
    for j in range(len(total_spend_order)):
        text = ax[1].text(
            j, i, other_heatmap_data[i, j], ha="center", va="center", color="w"
        )

ax[1].set_title("Monthly Total Coffee Spend by Working Condition")
ax[1].set_xlabel("Amount spent on Coffee Monthly")
ax[1].set_ylabel("Working conditions")
fig.tight_layout()
plt.show()


# We can see that:<br/>
# + people who work from home tend to spend the most on coffee equipment (~ 25% spend more than $1000).<br/>
# + people who work from the office or hybrid also tend to typically spend about $100-300 on equipment.<br/>
# + The `Unemployed` category has the most varied spend, but this can be accounted for by the demographics (students might spend < $20, temporarily unemployed people or retired people might have already invested in equipment previously which would explain the higher spend in $100-$300 and $500-$1000 range)<br/>
# Overall, people tend to spend a significant amount of money on coffee equipment regardless of their working conditions
# <br/><br/>
# In the second graph, we can see that, in general people spend <$60 monthly on coffee even if they have spent a lot of coffee equipment<br/>
# 
# 

# In[35]:


generation_order = ["Gen Z", "Millennial", "Gen X", "Boomer"]

cups_order = [
    "Less than 1",
    "1",
    "2",
    "3",
    "4",
    "More than 4"
]

#values = new_df.groupby(by=["age", "cups"])[['age', 'cups']].value_counts()
#new_index = pd.MultiIndex.from_product(
#    [age_order, cups_order], names=["Age group", "Daily cups of coffee"]
#)
#values = values.reindex(new_index).fillna(0).to_numpy().reshape((7,6))
values = pd.crosstab(new_df["generation"], new_df["total_spend"], normalize='index')*100
values = values.reindex(index=generation_order, columns=total_spend_order).fillna(0).to_numpy()

fig, ax = plt.subplots()
ax.set_xticks(range(len(generation_order)), labels=generation_order, rotation=45)
#ax.set_yticks(range(len(cups_order)), labels=cups_order)
bottom_values = np.zeros(len(generation_order))

for i, subcategory in enumerate(total_spend_order):
    ax.bar(generation_order, values[:, i], bottom=bottom_values, label=subcategory)
    bottom_values += values[:, i]

ax.set_ylabel("Percentage(%)")
ax.set_xlabel('Generation')
ax.set_title("Coffee Drinkers by Generation & Monthly Spend")
ax.legend(title="Monthly Spend", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.show()


# We can see that ~30% `Gen Z` people tend to spend `$20-$40`, while `~15%` spend `$40-$60`, but this ratio moves in the other direction with age, maybe because people have more disposable income?
# It goes back to the initial observation for `Boomer`, might be because older people give up on drinking coffee as often

# In[641]:


values = new_df.groupby(by=["generation", "cups"])[
    ["generation", "cups"]
].value_counts()
new_index = pd.MultiIndex.from_product(
   [generation_order, cups_order], names=["Generation", "Daily cups of coffee"] 
)
# values = values.reindex(new_index).fillna(0).to_numpy().reshape((7,6))
values = values.reindex(new_index).fillna(0).reset_index(name="Count")

fig, ax = plt.subplots(figsize=(6, 5))

bubble = ax.scatter(
    x=values["Generation"],
    y=values["Daily cups of coffee"],
    s=values["Count"] * 25,  # Control marker size (area)
    c=values["Count"],  # Map bubble color to values
    cmap="viridis",
    alpha=0.75,  # Blend overlapping markers
    #edgecolors="black",
)

# 4. Add details and colorbar
ax.set_xticks(range(len(generation_order)), labels=generation_order, rotation=45)
ax.set_title("Coffee Drinkers by Generation & Volume")
ax.set_xlabel("Generation")
ax.set_ylabel("Daily cups of coffee")
ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)  # Move gridlines behind the bubbles

# Add colorbar indicator
cbar = fig.colorbar(bubble, ax=ax, shrink=0.8)
cbar.set_label("Frequency", rotation=270, labelpad=15)

plt.show()


# In[642]:


top_5_coffees = ["Pourover", "Latte", "Regular drip coffee", "Cappuccino", "Espresso"]
genders = ["Male", "Female"]

filtered_df = new_df[new_df["favorite"].isin(top_5_coffees) & new_df["gender"].isin(genders)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), sharey=True)
x_indices = np.arange(len(generation_order))  
bar_width = 0.15                      
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # 5 distinct colors

for ax_idx, gender in enumerate(genders):
    ax = axes[ax_idx]

    # Filter dataset for the current gender
    gender_df = filtered_df[filtered_df['gender'] == gender]

    crosstab = pd.crosstab(gender_df['generation'], gender_df['favorite'], normalize='index')*100
    crosstab = crosstab.reindex(index=generation_order, columns=top_5_coffees).fillna(0)

    # Plot each coffee type as a shifted bar
    for coffee_idx, coffee in enumerate(top_5_coffees):
        # Shift the X position of each bar so they line up sequentially
        shifted_x = x_indices + (coffee_idx * bar_width) - (len(top_5_coffees) * bar_width / 2) + (bar_width / 2)

        ax.bar(
            shifted_x, 
            crosstab[coffee], 
            width=bar_width, 
            label=coffee if ax_idx == 0 else "",
            color=colors[coffee_idx],
            edgecolor='black',
            linewidth=0.5
        )

    # Formatting specific to each subplot panel
    ax.set_xticks(x_indices)
    ax.set_xticklabels(generation_order)
    ax.set_xlabel("Age Group", fontsize=11, labelpad=8)
    ax.set_title(f"{gender} Respondents", fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

# 5. Global Layout Styling & Legends
axes[0].set_ylabel("Number of Drinkers", fontsize=11, labelpad=8)

# Place a single shared clean legend box completely outside the graph margins
fig.legend(title="Favorite Coffee Type", loc='center right', bbox_to_anchor=(1.12, 0.5), title_fontsize=11)

fig.suptitle("Top 5 Favorite Coffee Types by Age Group and Gender", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[ ]:




