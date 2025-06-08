import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Display options for better readability
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Load the Titanic dataset
titanic_df = sns.load_dataset('titanic')

# Dataset Overview
print(f"Dataset Shape: {titanic_df.shape[0]} rows and {titanic_df.shape[1]} columns")
print("\nFirst 5 rows of the dataset:")
print(titanic_df.head())

print("\nData Types:")
print(titanic_df.dtypes)

print("\nMissing Values:")
missing = titanic_df.isnull().sum()
missing_percent = (missing / len(titanic_df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing,
                             'Percentage': missing_percent})
print(missing_data[missing_data['Missing Values'] > 0])

print("\nSummary Statistics for Numerical Features:")
print(titanic_df.describe())

print("\nSummary Statistics for Categorical Features:")
print(titanic_df.describe(include=['O']))

# Getting the survival rate
print(titanic_df['survived'].value_counts())
print(f"Survival Rate: {titanic_df['survived'].mean() * 100:.2f}%")

# Survival distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='survived', data=titanic_df, palette='RdYlGn')
plt.title('Survival Distribution')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
for i in [0, 1]:
    count = len(titanic_df[titanic_df['survived'] == i])
    pct = count / len(titanic_df) * 100
    plt.text(i, count + 10, f'{count} ({pct:.1f}%)', ha='center')
plt.show()

# Survival rate by passenger class
plt.figure(figsize=(10, 6))
sns.countplot(x='pclass', hue='survived', data=titanic_df, palette='RdYlGn')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
for i in range(1, 4):
    total = len(titanic_df[titanic_df['pclass'] == i])
    survived = len(titanic_df[(titanic_df['pclass'] == i) & (titanic_df['survived'] == 1)])
    rate = survived / total * 100
    plt.text(i-1, total/2, f'Survival Rate: {rate:.1f}%', ha='center')
plt.show()

# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=titanic_df, x='age', hue='survived', multiple='stack', bins=20, palette='RdYlGn')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.axvline(titanic_df['age'].mean(), color='red', linestyle='--', label=f'Mean Age: {titanic_df["age"].mean():.1f}')
plt.legend(title='Survived', labels=['No', 'Yes', 'Mean Age'])
plt.show()

# Survival rate by gender
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', hue='survived', data=titanic_df, palette='RdYlGn')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
for i, gender in enumerate(['male', 'female']):
    total = len(titanic_df[titanic_df['sex'] == gender])
    survived = len(titanic_df[(titanic_df['sex'] == gender) & (titanic_df['survived'] == 1)])
    rate = survived / total * 100
    plt.text(i, total/2, f'Survival Rate: {rate:.1f}%', ha='center')
plt.show()

# Survival rate by embarked port
plt.figure(figsize=(10, 6))
sns.countplot(x='embarked', hue='survived', data=titanic_df, palette='RdYlGn')
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
for i, port in enumerate(['C', 'Q', 'S']):
    if port in titanic_df['embarked'].unique():  # Check if port exists in data
        total = len(titanic_df[titanic_df['embarked'] == port])
        survived = len(titanic_df[(titanic_df['embarked'] == port) & (titanic_df['survived'] == 1)])
        rate = survived / total * 100
        plt.text(i, total/2, f'Survival Rate: {rate:.1f}%', ha='center')
plt.show()

# Survival rate by family size (siblings/spouse + parents/children)
titanic_df['family_size'] = titanic_df['sibsp'] + titanic_df['parch']
plt.figure(figsize=(12, 6))
sns.countplot(x='family_size', hue='survived', data=titanic_df, palette='RdYlGn')
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
for i in range(0, titanic_df['family_size'].max() + 1):
    if i in titanic_df['family_size'].unique():  # Check if family size exists
        total = len(titanic_df[titanic_df['family_size'] == i])
        survived = len(titanic_df[(titanic_df['family_size'] == i) & (titanic_df['survived'] == 1)])
        rate = survived / total * 100
        if total > 10:  # Only show percentage for bars with sufficient data
            plt.text(i, total/2, f'{rate:.1f}%', ha='center')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
numeric_df = titanic_df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='RdYlGn', fmt='.2f', mask=mask)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Fare distribution by passenger class and survival
plt.figure(figsize=(12, 6))
sns.boxplot(x='pclass', y='fare', hue='survived', data=titanic_df, palette='RdYlGn')
plt.title('Fare Distribution by Passenger Class and Survival')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Age vs Fare with survival and passenger class
plt.figure(figsize=(12, 10))
sns.scatterplot(x='age', y='fare', hue='survived', size='pclass',
                sizes=(100, 50), alpha=0.7, palette='RdYlGn', data=titanic_df)
plt.title('Age vs Fare with Survival and Passenger Class')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Survivability based on age group
titanic_df['age_group'] = pd.cut(titanic_df['age'], bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

plt.figure(figsize=(12, 6))
sns.countplot(x='age_group', hue='survived', data=titanic_df, palette='RdYlGn')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
for i, age_group in enumerate(titanic_df['age_group'].unique()):
    if not pd.isna(age_group) and i < len(titanic_df['age_group'].dropna().unique()):
        total = len(titanic_df[titanic_df['age_group'] == age_group])
        survived = len(titanic_df[(titanic_df['age_group'] == age_group) & (titanic_df['survived'] == 1)])
        rate = survived / total * 100
        plt.text(i, total/2, f'{rate:.1f}%', ha='center')
plt.show()

# Gender, Class and Age interaction with Survival
plt.figure(figsize=(16, 10))
g = sns.FacetGrid(titanic_df, col='sex', row='pclass', margin_titles=True, height=4)
g.map(sns.histplot, 'age', hue='survived', multiple='stack', bins=20)
g.add_legend(title='Survived', labels=['No', 'Yes'])
g.fig.suptitle('Age Distribution by Gender, Class, and Survival', y=1.05, fontsize=20)
plt.show()

# Interactive survival distribution
fig = px.histogram(titanic_df, x="survived",
                   color_discrete_sequence=['skyblue'],
                   labels={"survived": "Survived (0 = No, 1 = Yes)"},
                   title="Survival Distribution",
                   category_orders={"survived": [0, 1]})
fig.update_layout(bargap=0.2)
fig.show()

# Interactive survival by class
fig = px.histogram(titanic_df, x="pclass", color="survived",
                  barmode="group",
                  color_discrete_sequence=['darkblue', 'skyblue'],
                  title="Survival by Passenger Class",
                  labels={"pclass": "Passenger Class", "survived": "Survived"},
                  category_orders={"survived": [0, 1]})
fig.update_layout(bargap=0.2)
fig.show()

# Interactive age distribution
fig = px.histogram(titanic_df, x="age", color="survived",
                  marginal="box",
                  color_discrete_sequence=['darkblue', 'skyblue'],
                  title="Age Distribution by Survival Status",
                  labels={"age": "Age", "survived": "Survived"},
                  category_orders={"survived": [0, 1]},
                  nbins=20)
fig.update_layout(bargap=0.1)
fig.show()

# Interactive fare distribution
fig = px.box(titanic_df, x="pclass", y="fare", color="survived",
             color_discrete_sequence=['darkblue', 'skyblue'],
             title="Fare Distribution by Class and Survival",
             labels={"pclass": "Passenger Class", "fare": "Fare", "survived": "Survived"},
             category_orders={"survived": [0, 1], "pclass": [1, 2, 3]})
fig.show()

# Interactive scatter plot: Age vs Fare
fig = px.scatter(titanic_df, x="age", y="fare", color="survived", size="pclass",
                size_max=15, opacity=0.7,
                color_discrete_sequence=['darkblue', 'skyblue'],
                title="Age vs Fare with Survival and Passenger Class",
                labels={"age": "Age", "fare": "Fare", "survived": "Survived", "pclass": "Passenger Class"},
                category_orders={"survived": [0, 1], "pclass": [1, 2, 3]})
fig.update_layout(legend_title_text='Legend')
fig.show()

# Interactive family size analysis
fig = px.histogram(titanic_df, x="family_size", color="survived",
                  barmode="group",
                  color_discrete_sequence=['darkblue', 'skyblue'],
                  title="Survival by Family Size",
                  labels={"family_size": "Family Size", "survived": "Survived"},
                  category_orders={"survived": [0, 1]})
fig.update_layout(bargap=0.2)
fig.show()

# Interactive correlation heatmap
fig = px.imshow(correlation,
               color_continuous_scale='Blues',
               title="Correlation Heatmap of Numerical Features")
fig.update_layout(width=800, height=800)
fig.show()

# Interactive survival by port
fig = px.histogram(titanic_df.dropna(subset=['embarked']), x="embarked", color="survived",
                  barmode="group",
                  color_discrete_sequence=['darkblue', 'skyblue'],
                  title="Survival by Port of Embarkation",
                  labels={"embarked": "Port of Embarkation", "survived": "Survived"},
                  category_orders={"survived": [0, 1], "embarked": ["C", "Q", "S"]})
fig.update_layout(bargap=0.2)
fig.show()

# Interactive survival by title
fig = px.histogram(titanic_df, x="title", color="survived",
                  barmode="group",
                  color_discrete_sequence=['darkblue', 'skyblue'],
                  title="Survival by Title",
                  labels={"title": "Title", "survived": "Survived"},
                  category_orders={"survived": [0, 1]})
fig.update_layout(bargap=0.2)
fig.show()

# Interactive survival by age group
fig = px.histogram(titanic_df.dropna(subset=['age_group']), x="age_group", color="survived",
                  barmode="group",
                  color_discrete_sequence=['darkblue', 'skyblue'],
                  title="Survival by Age Group",
                  labels={"age_group": "Age Group", "survived": "Survived"},
                  category_orders={"survived": [0, 1],
                                   "age_group": ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']})
fig.update_layout(bargap=0.2)
fig.show()

# Survival Rate Dashboard using Plotly subplots
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=['Survival by Class', 'Survival by Gender',
                                    'Survival by Age Group', 'Survival by Embarkation Port'])

# Survival by Class subplot
class_data = titanic_df.groupby('pclass')['survived'].mean().reset_index()
fig.add_trace(go.Bar(x=class_data['pclass'], y=class_data['survived']*100,
                     name='By Class', marker_color='skyblue'), row=1, col=1)

# Survival by Gender subplot
gender_data = titanic_df.groupby('sex')['survived'].mean().reset_index()
fig.add_trace(go.Bar(x=gender_data['sex'], y=gender_data['survived']*100,
                     name='By Gender', marker_color='darkblue'), row=1, col=2)

# Survival by Age Group subplot
age_group_data = titanic_df.groupby('age_group')['survived'].mean().reset_index()
fig.add_trace(go.Bar(x=age_group_data['age_group'], y=age_group_data['survived']*100,
                     name='By Age Group', marker_color='royalblue'), row=2, col=1)