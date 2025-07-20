
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


try:
    df = pd.read_csv("train.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'train.csv' not found in script directory")
    exit()

print("\nOriginal columns:", df.columns.tolist())



if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop('Cabin', axis=1, inplace=True, errors='ignore')  


if all(col in df.columns for col in ['SibSp', 'Parch']):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
else:
    print("\nWarning: Could not create FamilySize - SibSp/Parch columns missing")


plt.figure(figsize=(18, 12))


plt.subplot(2, 3, 1)
sns.countplot(x='Survived', data=df, palette='viridis')
plt.title("Survival Count (0=Died, 1=Survived)")


plt.subplot(2, 3, 2)
if 'Pclass' in df.columns:
    sns.barplot(x='Pclass', y='Survived', data=df, palette='magma')
    plt.title("Survival by Passenger Class")
else:
    plt.text(0.5, 0.5, 'Pclass data missing', ha='center')


plt.subplot(2, 3, 3)
if 'Sex' in df.columns:
    sns.countplot(x='Sex', hue='Survived', data=df, palette='coolwarm')
    plt.title("Survival by Gender")
else:
    plt.text(0.5, 0.5, 'Sex data missing', ha='center')


plt.subplot(2, 3, 4)
if 'Age' in df.columns:
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title("Age Distribution")
else:
    plt.text(0.5, 0.5, 'Age data missing', ha='center')


plt.subplot(2, 3, 5)
if 'Fare' in df.columns:
    sns.boxplot(x='Pclass', y='Fare', data=df, palette='rocket')
    plt.title("Fare by Class")
else:
    plt.text(0.5, 0.5, 'Fare data missing', ha='center')


plt.subplot(2, 3, 6)
if 'FamilySize' in df.columns:
    sns.barplot(x='FamilySize', y='Survived', data=df, palette='Set2')
    plt.title("Survival by Family Size")
elif 'Parch' in df.columns:  
    sns.barplot(x='Parch', y='Survived', data=df, palette='Set2')
    plt.title("Survival by Parents/Children")
else:
    plt.text(0.5, 0.5, 'Family data unavailable', ha='center')

plt.tight_layout()
plt.show() 
plt.savefig('titanic_results.png')
print("\nVisualization saved as 'titanic_results.png'")


print("\n=== Key Statistics ===")
if 'Survived' in df.columns:
    print(f"Overall survival: {df['Survived'].mean():.1%}")
    
    if 'Sex' in df.columns:
        print(f"Female survival: {df[df['Sex']=='female']['Survived'].mean():.1%}")
    
    if 'Pclass' in df.columns:
        print(f"1st Class survival: {df[df['Pclass']==1]['Survived'].mean():.1%}")


df.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned data saved as 'titanic_cleaned.csv'")
