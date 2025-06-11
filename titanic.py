# 1. Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# 2. Load the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# 3. Handling missing data(Data cleaning)
train['Age'].fillna(train['Age'].mean(), inplace = True)
train['Embarked'].fillna('S' , inplace = True)
test['Age'].fillna(train['Age'].mean(), inplace = True)
test['Fare'].fillna(train['Fare'].mean() , inplace = True)


# 4. Convert categorical columns to numbers(Encoding)
train['Sex'] = train['Sex'].map({'male' : 0, 'female' : 1})
train['Embarked'] = train['Embarked'].map({'S':0, 'C':1, 'Q' : 2})
test['Sex'] = test['Sex'].map({'male':0 ,'female' : 1})
test['Embarked'] = test['Embarked'].map({'S' : 0, 'C' : 1, 'Q':2})

                        
# 5. Data visualization

# Plot 1: heatmap to see where data is missing
plt.figure(figsize = (8,5))
sns.heatmap(train.isnull(), cbar = False, cmap = 'viridis')
plt.title("Missing values in Dataset")
plt.show()

# Plot 2: Count how many survived vs didn't survive
survived_counts = train['Survived'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(['Did not Survive', 'Survived'], survived_counts, color='lightpink', edgecolor='black')
plt.title("Survival Count", color='deeppink')
plt.xlabel("Survival Status", color='deeppink')
plt.ylabel("Passenger Count", color='deeppink')
plt.show()

# Plot 3: Survival count by Gender
gender_counts = train['Sex'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(gender_counts, labels=['Male', 'Female'], colors=['hotpink', 'skyblue'],
        autopct='%1.1f%%', startangle=90, textprops={'color':'black'})
plt.title("Passenger Gender Distribution")
plt.axis('equal')  # Makes it a perfect circle
plt.show()


# Plot 4: Bar chart - Survival count by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=train, palette=['lightpink', 'deeppink'])
plt.title("Survival by Passenger Class", color='deeppink')
plt.xlabel("Passenger Class (1 = High, 3 = Low)")
plt.ylabel("Passenger Count")
plt.legend(["Did not Survive", "Survived"], title="Survival", loc='upper right')
plt.show()


#plot 5 : Age Distribution histogram
plt.figure(figsize = (8,5))
train['Age'].plot.hist(bins = 30, edgecolor = 'black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# Plot 6 : Boxplot Showing age Vs Passenger Class
sns.boxplot(x = 'Pclass' , y = 'Age', data = train,color = 'yellow')
plt.title("Age by Passenger Class")
plt.show()


# 6. Model Building - Prepare data for training
features = ['Pclass' , 'Sex' , 'Age' , 'Fare' , 'Embarked']
x = train[features]
y = train['Survived']


# Split dataset into trainig and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2 , random_state = 42)


# Create and train random forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)


#check model accuracy on validation set
val_score = model.score(x_val, y_val)
print("Validation Accuracy:" , round(val_score*100, 2), "%")


# 7. Predict on test Data
X_test = test[features]
predictions = model.predict(X_test)


# 8. Savepredictions in submission file
submission = pd.DataFrame({
    'PassengerID': test['PassengerId'],
    'Survived' : predictions
    })
submission.to_csv('submission.csv',index = False)
print("✅ submission.csv file saved successfully")
