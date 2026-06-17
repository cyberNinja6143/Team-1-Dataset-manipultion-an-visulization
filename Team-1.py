#!/usr/bin/env python
# coding: utf-8

# In[264]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Loading the data

# In[265]:


df = pd.read_csv('24_Coffee_Taste_Test.csv')
df.head()

# Clean the data

# Remove any trailing/leading spaces in text columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.strip()

# Remove rows that have nothing but NA
is_dead_row = df.iloc[:, 1:].isin(["NA", "", None]).all(axis=1)
df = df[~is_dead_row]

# Remove malicious code, this is best practice even if the data soruce is trustworthy
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = (
            df[col]
            .str.replace("<script>", "", case=False, regex=False)
            .str.replace("</script>", "", case=False, regex=False)
            .str.replace("javascript:", "", case=False, regex=False)
        )


# # Raw data overview
# ## Shape and Data types
# The dataset has 4042 rows and 57 columns of which 13 are floats and 44 are objects.

# In[266]:


df.shape


# In[267]:


df.info()


# ## Null Counts

# In[268]:


df.isnull().sum().sort_values()


# ## Basic Statistics

# In[269]:


float_cols = df.select_dtypes(include=["float"]).columns.tolist()
df[float_cols].describe()


# ## Interesting columns
# ### favorite
# This column denotes the subject's favorite type of coffee 

# In[270]:


df.favorite.value_counts(dropna=False)


# ### gender
# This column denotes the subject's gender

# In[271]:


df.gender.value_counts(dropna=False)


# ### total_spend
# This column denotes the range of money the subject spends on coffee per month

# In[272]:


df.total_spend.value_counts(dropna=False)


# ## age
# This column denotes the age bracket of the subject

# In[273]:


df.age.value_counts(dropna=False)


# ### where_drink
# This column denotes where( home, at a cafe etc..) people prefer to drink coffee

# In[274]:


df.where_drink.value_counts(dropna=False)


# On initial inspection, the column data seems to be a multiple choice answer on the survey. We can observe some consistent categories here like:
# + At home
# + At a cafe
# + At the office
# + On the go

# ### wfh
# This column denotes whether the subject works from home / office.

# In[275]:


df.wfh.value_counts(dropna=False)


# This column has a lot of missing values, but would be interesting to analyze with the column `where_drink`

# ### spent_equipment
# This column denotes how much a subject has spent on equipment to brew coffee

# In[276]:


df.spent_equipment.value_counts(dropna=False)


# ### cups
# This column denotes the average cups of a coffee a subject drinks in a day.

# In[277]:


df.cups.value_counts(dropna=False)


# In[278]:


df.employment_status.value_counts(dropna=False)


# In[279]:


df[df["wfh"].isna() & df["employment_status"].notna()][["wfh", "employment_status"]]


# To conclude, the data has a lot of missing values, but it does not suffer from inconsistent formatting, outliers or duplicate values. So the only cleaning required is deciding what to do with the missing values

# # Cleaning the data

# ### favorite
# This column has only 62 missing values and over 10 categories.<br/> 
# It would not make sense to impute a value here, since we cannot be certain what a subject's favorite coffee is.<br/>
# Because there are very few missing values, we can safely drop them.

# In[280]:


df = df.dropna(subset=["favorite"])
df.favorite.value_counts(dropna=False)


# ### gender
# This column has over 500 missing values.<br/>
# However, we can reasonably assume that the people who did not specify a gender, preferred not to share their gender. <br/>
# So we fill the nan values in this column with `Prefer not to say`

# In[281]:


df['gender'] = df['gender'].fillna('Prefer not to say')
df.gender.value_counts(dropna=False)


# ### total_spend
# For now labelling nans as unknown

# In[263]:


df = df.dropna(subset=["total_spend"])
df.total_spend.value_counts(dropna=False)


# ### age
# This column has very few missing values, and there is no reliable way to ascertain the age group for a subject, so we drop them.

# In[282]:


df = df.dropna(subset=["age"])
df.age.value_counts(dropna=False)


# ### wfh
# The category names are very long, so we shorten them to `Work from home`, `Work from office` and `Hybrid`

# In[283]:


wfh_mapping = {
    "I primarily work from home": "Work from home",
    "I primarily work in person": "Work from office",
    "I do a mix of both": "Hybrid",
}

# Force Pandas to modify the column safely in place
df.loc[:, "wfh"] = df["wfh"].map(wfh_mapping).fillna(df["wfh"])


# In[284]:


df.wfh.value_counts(dropna=False)


# This column has a significant amount of missing values.<br/>
# We can impute some values using the `employment_status` column

# In[285]:


df.employment_status.value_counts(dropna=False)


# We can assume that people who have `employment_status` value as `Unemployed`, `Retired`, `Student` do not work so we can create a new status for them as `Unemployed`.<br/>
# People with the `Homemaker` status are assumed to be working from home.

# In[286]:


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


# In[287]:


# For now fill nan as Unknown
df = df.dropna(subset=["wfh"])
df.wfh.value_counts(dropna=False)


# ### spent_equipment

# In[288]:


df = df.dropna(subset=["spent_equipment"])
df.spent_equipment.value_counts(dropna=False)


# ### cups

# In[289]:


df = df.dropna(subset=["cups"])
df.cups.value_counts(dropna=False)


# In[290]:


new_df = df[['favorite', 'gender', 'total_spend', 'age', 'wfh', 'spent_equipment', 'cups']]
new_df.head()


# In[291]:


new_df.shape


# # Visualizations 
# 

# In[292]:


pd.crosstab(new_df["wfh"], new_df["spent_equipment"], normalize="index") * 100


# In[333]:


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
    "$40-$600",
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


# In[ ]:


age_order = [
    "<18 years old",
    "18-24 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    ">65 years old"
]

cups_order = [
    "Less than 1",
    "1",
    "2",
    "3",
    "4",
    "More than 4"
]

values = new_df.groupby(by=["age", "cups"])[['age', 'cups']].value_counts()
#new_index = pd.MultiIndex.from_product(
#    [age_order, cups_order], names=["Age group", "Daily cups of coffee"]
#)
#values = values.reindex(new_index).fillna(0).to_numpy().reshape((7,6))
values = pd.crosstab(new_df["age"], new_df["cups"])
values = values.reindex(index=age_order, columns=cups_order).fillna(0).to_numpy()

fig, ax = plt.subplots()
ax.set_xticks(range(len(age_order)), labels=age_order, rotation=45)
#ax.set_yticks(range(len(cups_order)), labels=cups_order)
bottom_values = np.zeros(len(age_order))

for i, subcategory in enumerate(cups_order):
    ax.bar(age_order, values[:, i], bottom=bottom_values, label=subcategory)
    bottom_values += values[:, i]

ax.set_ylabel("Values")
ax.set_xlabel('Age group')
ax.set_title("Coffee Drinkers by Age & Volume")
ax.legend(title="Subcategories")

plt.show()


# In[336]:


values = new_df.groupby(by=["age", "cups"])[["age", "cups"]].value_counts()
new_index = pd.MultiIndex.from_product(
    [age_order, cups_order], names=["Age group", "Daily cups of coffee"]
)
# values = values.reindex(new_index).fillna(0).to_numpy().reshape((7,6))
values.reindex(new_index).fillna(0).reset_index(name='Count')


# In[341]:


values = new_df.groupby(by=["age", "cups"])[["age", "cups"]].value_counts()
new_index = pd.MultiIndex.from_product(
   [age_order, cups_order], names=["Age group", "Daily cups of coffee"] 
)
# values = values.reindex(new_index).fillna(0).to_numpy().reshape((7,6))
values = values.reindex(new_index).fillna(0).reset_index(name="Count")

fig, ax = plt.subplots(figsize=(6, 5))

bubble = ax.scatter(
    x=values["Age group"],
    y=values["Daily cups of coffee"],
    s=values["Count"] * 25,  # Control marker size (area)
    c=values["Count"],  # Map bubble color to values
    cmap="viridis",
    alpha=0.75,  # Blend overlapping markers
    #edgecolors="black",
)

# 4. Add details and colorbar
ax.set_xticks(range(len(age_order)), labels=age_order, rotation=45)
ax.set_title("Coffee Drinkers by Age & Volume")
ax.set_xlabel("Age group")
ax.set_ylabel("Daily cups of coffee")
ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)  # Move gridlines behind the bubbles

# Add colorbar indicator
cbar = fig.colorbar(bubble, ax=ax, shrink=0.8)
cbar.set_label("Frequency", rotation=270, labelpad=15)

plt.show()


# # Ignore

# In[ ]:


df.additions.value_counts(dropna=False)


# In[ ]:


def explode_col(df, column_name: str):
    # 1. Force everything to string, split, and explode
    df[column_name] = df[column_name].astype(str).str.split(r"\s*,\s*")
    df = df.explode(column_name)

    # 2. Convert literal 'nan' strings back to actual NaNs
    df[column_name] = df[column_name].replace("nan", np.nan)
    return df

where_drink_exploded_df = explode_col(df, "where_drink")


# In[67]:


where_drink_exploded_df.shape


# In[68]:


where_drink_exploded_df.where_drink.value_counts(dropna=False)


# In[91]:


df.wfh.value_counts(dropna=False)


# In[83]:


place_df = where_drink_exploded_df.groupby(by=["wfh", "where_drink"], as_index=False)[
    ["wfh", "where_drink"]
].value_counts()


# In[ ]:


place_df = where_drink_exploded_df.groupby(by=["wfh", "where_drink"], as_index=False)[
    ["wfh", "where_drink"]
].value_counts()

place_df["percentage"] = place_df.groupby("wfh")["count"].transform(
    lambda x: (x / x.sum()) * 100
)

plot_data = place_df.pivot(index="wfh", columns="where_drink", values="percentage")

plot_data.plot(kind="bar", stacked=True, figsize=(8, 6), edgecolor="black")

plt.title("Where People Drink Based on WFH Status (Normalized %)")
plt.ylabel("Percentage (%)")
plt.xlabel("WFH Status")
plt.legend(title="Where They Drink", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[ ]:




