# In[1]:


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)



# In[2]:


import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline


# # Prepare Data

# In[3]:


get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:////home/jovyan/nepal.sqlite')


# In[4]:


get_ipython().run_cell_magic('sql', '', 'SELECT distinct(district_id)\nFROM id_map')


# In[6]:


get_ipython().run_cell_magic('sql', '', 'SELECT count(*)\nFROM id_map\nWHERE district_id = 1')


# In[8]:


get_ipython().run_cell_magic('sql', '', 'SELECT count(*)\nFROM id_map\nWHERE district_id = 3')

# In[10]:


get_ipython().run_cell_magic('sql', '', 'SELECT distinct(i.building_id) AS b_id, s.*, d.damage_grade\nFROM id_map AS i\nJOIN building_structure AS s ON i.building_id = s.building_id\nJOIN building_damage AS d ON i.building_id = d.building_id\nWHERE district_id=3\nLIMIT 5')


# ## Import

# In[12]:


def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
        SELECT distinct(i.building_id) AS b_id, s.*, d.damage_grade
        FROM id_map AS i
        JOIN building_structure AS s ON i.building_id = s.building_id
        JOIN building_damage AS d ON i.building_id = d.building_id
        WHERE district_id=3
        """

    # Read query results into DataFrame
    df = pd.read_sql(query, conn, index_col="b_id")

    # Identify leaky columns
    drop_cols=[col for col in df.columns if "post_eq" in col]

    # Create binany target
    df["damage_grade"] = df["damage_grade"].str[-1].astype(int)
    df["severe_damage"] = (df["damage_grade"] > 3).astype(int)

    # Drop old target
    drop_cols.append("damage_grade")

    # Drop multicollinearity
    drop_cols.append("count_floors_pre_eq")

    # Drop high-cardinality categorical columns
    drop_cols.append("building_id")

    # Drop columns
    df.drop(columns=drop_cols, inplace=True)

    return df


# In[13]:


df = wrangle("/home/jovyan/nepal.sqlite")
df.head()


# ## Explore

# In[16]:


# Plot value counts of `"severe_damage"`
df["severe_damage"].value_counts(normalize=True).plot(
    kind="bar", xlabel="Severe Damage", ylabel="Relative Frequency", title="Kavrepalanchok, Class Balance"
);


# In[19]:


sns.boxplot(x="severe_damage", y="plinth_area_sq_ft", data=df)
plt.xlabel("Severe Damage")
plt.ylabel("Plinth Area [sq. ft.]")
plt.title("Kavrepalanchok, Plinth Area vs Building Damage");
# Don't delete the code below 👇
plt.savefig("images/4-5-7.png", dpi=150)


# In[21]:


roof_pivot = pd.pivot_table(
    df, index="roof_type", values="severe_damage", aggfunc=np.mean
)
roof_pivot


# In[25]:


target = "severe_damage"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[27]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# # Build Model

# ## Baseline

# In[29]:


acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))


# ## Iterate

# In[31]:


model_lr = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)
model_lr.fit(X_train, y_train)


# In[33]:


lr_train_acc = model_lr.score(X_train, y_train)
lr_val_acc = model_lr.score(X_val, y_val)

print("Logistic Regression, Training Accuracy Score:", lr_train_acc)
print("Logistic Regression, Validation Accuracy Score:", lr_val_acc)


# In[35]:


depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []
for d in depth_hyperparams:
    model_dt = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=d, random_state=42)
    )
    model_dt.fit(X_train, y_train)
    training_acc.append(model_dt.score(X_train, y_train))
    validation_acc.append(model_dt.score(X_val, y_val))


# In[37]:


plt.plot(depth_hyperparams, training_acc, label="training")
plt.plot(depth_hyperparams, validation_acc, label="validation")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.title("Validation Curve, Decision Tree Model")
plt.legend();


# In[38]:


plt.plot(depth_hyperparams, training_acc, label="training")
plt.plot(depth_hyperparams, validation_acc, label="validation")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.title("Validation Curve, Decision Tree Model")
plt.legend();


# ## Evaluate

# In[46]:


X_test = pd.read_csv("data/kavrepalanchok-test-features.csv", index_col="b_id")
y_test_pred = final_model_dt.predict(X_test)
y_test_pred[:5]


